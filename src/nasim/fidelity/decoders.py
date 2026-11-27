from __future__ import annotations

import numpy as np
import stim
from ldpc import BpOsdDecoder
import pymatching

def dem_to_check_matrices(dem: stim.DetectorErrorModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Lowers a Stim DetectorErrorModel into the (check_matrix, observable_matrix,
    priors) triple a generic decoder (BP-OSD, etc.) needs.
    """

    num_detectors = dem.num_detectors
    num_observables = dem.num_observables

    check_cols = []
    obs_cols = []
    priors = []

    for instr in dem.flattened():
        if instr.type != "error": continue

        p = instr.args_copy()[0]
        dets, obs = [], []
        for tgt in instr.targets_copy():
            if tgt.is_relative_detector_id(): dets.append(tgt.val)
            elif tgt.is_logical_observable_id(): obs.append(tgt.val)

        col = np.zeros(num_detectors, dtype=np.uint8)
        col[dets] = 1
        check_cols.append(col)

        ocol = np.zeros(num_observables, dtype=np.uint8)
        ocol[obs] = 1
        obs_cols.append(ocol)

        priors.append(p)

    check_matrix = np.array(check_cols, dtype=np.uint8).T
    obs_matrix = np.array(obs_cols, dtype=np.uint8).T
    priors = np.array(priors, dtype=np.float64)

    return check_matrix, obs_matrix, priors

def decode_batch_bposd(dem: stim.DetectorErrorModel, detection_events: np.ndarray,
                       *, max_iter: int = 300, osd_order: int = 10, bp_method: str = "minimum_sum") -> np.ndarray:

    check_matrix, observable_matrix, priors = dem_to_check_matrices(dem)
    decoder = BpOsdDecoder(
        check_matrix,
        error_channel=list(priors),
        max_iter=max_iter,
        bp_method=bp_method,
        schedule="parallel",
        osd_method="OSD_0" if osd_order == 0 else "OSD_CS",
        osd_order=osd_order
    )

    shots = detection_events.shape[0]
    predicted = np.zeros((shots, observable_matrix.shape[0]), dtype=np.uint8)
    for i in range(shots):
        pred_errors = decoder.decode(detection_events[i].astype(np.uint8))
        predicted[i] = (observable_matrix @ pred_errors) % 2

    return predicted

def decode_batch_matching(dem: stim.DetectorErrorModel, detection_events: np.ndarray) -> np.ndarray:

    matcher = pymatching.Matching.from_detector_error_model(dem)
    return matcher.decode_batch(detection_events)

def build_gen_bicycle_xz_decoder(circuit: stim.Circuit, *, basis_coord: int, max_iter: int = 300,
                                 osd_order: int = 10, bp_method: str = "minimum_sum"):

    """
    One-time setup for decode_gen_bicycle_xz_split
    """

    dem = circuit.detector_error_model(decompose_errors=False)
    coords = dem.get_detector_coordinates()
    relevant = sorted(d for d, c in coords.items() if c and c[0] == basis_coord)
    det_index = {d: i for i, d in enumerate(relevant)}

    check_cols, obs_cols, priors = [], [], []
    for instr in dem.flattened():
        if instr.type != "error": continue

        p = instr.args_copy()[0]
        dets, obs = [], []
        for t in instr.targets_copy():
            if t.is_relative_detector_id(): dets.append(t.val)
            elif t.is_logical_observable_id(): obs.append(t.val)

        if not dets or any(d not in det_index for d in dets): continue

        col = np.zeros(len(relevant), dtype=np.uint8)
        for d in dets:
            col[det_index[d]] = 1
        check_cols.append(col)

        ocol = np.zeros(dem.num_observables, dtype=np.uint8)
        ocol[obs] = 1
        obs_cols.append(ocol)
        priors.append(p)

    check_matrix = np.array(check_cols, dtype=np.uint8).T
    obs_matrix = np.array(obs_cols, dtype=np.uint8).T
    priors = np.array(priors, dtype=np.float64)

    decoder = BpOsdDecoder(
        check_matrix,
        error_channel=list(priors),
        max_iter=max_iter,
        bp_method=bp_method,
        schedule="parallel",
        osd_method="OSD_0" if osd_order == 0 else "OSD_CS",
        osd_order=osd_order
    )

    relevant_arr = np.array(relevant, dtype=int)
    return decoder, relevant_arr, obs_matrix


def decode_with_prebuilt_decoder(decoder, relevant_arr: np.ndarray, obs_matrix: np.ndarray,
                                 detection_events: np.ndarray) -> np.ndarray:

    """Decode a batch of shots using a decoder already built by build_gen_bicycle_xz_decoder."""

    sub_events = detection_events[:, relevant_arr]
    shots = sub_events.shape[0]
    predicted = np.zeros((shots, obs_matrix.shape[0]), dtype=np.uint8)
    for i in range(shots):
        pred_errors = decoder.decode(sub_events[i].astype(np.uint8))
        predicted[i] = (obs_matrix @ pred_errors) % 2

    return predicted


def decode_gen_bicycle_xz_split(circuit: stim.Circuit, detection_events: np.ndarray, *, basis_coord: int, max_iter: int = 300,
                                osd_order: int = 10, bp_method: str = "minimum_sum"):

    """
    BP+OSD decoding doing X/Z in seq, following QS.
    """

    decoder, relevant_arr, obs_matrix = build_gen_bicycle_xz_decoder(
        circuit, basis_coord=basis_coord, max_iter=max_iter, osd_order=osd_order, bp_method=bp_method,
    )
    return decode_with_prebuilt_decoder(decoder, relevant_arr, obs_matrix, detection_events)