from __future__ import annotations

import math

def fit_power_law(p_values: list[float], rates: list[float]) -> tuple[float, float]:

    """
    Fit to rate(p) = Ap^n. Allow to test with values far below error tresh
    """

    log_p = [math.log10(p) for p in p_values]
    log_r = [math.log10(r) for r in rates]

    n_points = len(p_values)
    mean_x = sum(log_p) / n_points
    mean_y = sum(log_r) / n_points
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_p, log_r))
    var = sum((x - mean_x) ** 2 for x in log_p)

    n = cov / var
    log_A = mean_y - n * mean_x
    return 10.0 ** log_A, n

def predict_power_law(p: float, A: float, n: float) -> float:
    return A * (p ** n)

def r_squared(p_values: list[float], rates: list[float], A: float, n: float) -> float:

    log_r = [math.log10(r) for r in rates]
    log_pred = [math.log10(predict_power_law(p, A, n)) for p in p_values]
    mean_y = sum(log_r) / len(log_r)
    ss_tot = sum((y - mean_y) ** 2 for y in log_r)
    ss_res = sum((y - yp) ** 2 for y, yp in zip(log_r, log_pred))
    return 1.0 - ss_res / ss_tot