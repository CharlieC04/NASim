# Placement

NASim has two initial-placement algorithms.

| | `piqasso_placement` | `zap_placement` |
|---|---|---|
| Source | Piqasso (arXiv:2608.01316) | ZAP (Huang et al., IEEE TQE 2026), Eq. 6-11 |
| Device | any (uniform or zoned) | zoned only |
| Method | force-directed physics simulation | deterministic, single-pass, greedy |
| Randomness | seeded RNG + iteration | none |

## `piqasso_placement` - force-directed

Models qubits as particles in a 2D force field.

- **Spring attraction** between interacting qubit pairs, pulling them toward
  the ideal patch separation distance $d^*$, weighted by
  `circuit.interaction_weights()` - pairwise interaction strength decaying
  by stage index, $w = e^{-\delta \cdot \text{stage}}$, so qubits that
  interact *early* in the circuit are pulled together harder.
- **Inverse-square repulsion** between every pair.
- Qubits are **snapped** to the nearest free device site, processed
  in order of total interaction degree (most-connected qubits get first
  pick of their preferred site).

## `zap_placement` - deterministic, routing-aware

Built specifically for zoned devices, following ZAP Eq. 6-11:

1. **Rank qubits by priority.** $W(q) = \sum_{\ell} w(\ell)$, summed over
   every stage $\ell$ qubit $q$ appears in (any gate type), with
   $w(\ell) = 1/(\ell+1)$.
2. **For each qubit, in priority order**, score every free storage site $u$:
   $$
   \text{score}(u) = \lambda_{par} \cdot c(u) + d_{min}(u)
   $$
   where $d_{min}(u)$ is $u$'s distance to its nearest entanglement site
   (Eq. 6-7), and $c(u)$ counts how many *already-committed* qubits' first
   -move vectors (`source=u, target=`nearest entanglement site) conflict
   with this one under the router's own AOD compatibility rule (see
   [Movement & Routing](movement.md)).
3. **Commit the best site**, add its move vector to the committed set, move
   to the next qubit.