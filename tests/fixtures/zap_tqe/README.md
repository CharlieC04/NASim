# ZAP tqe benchmark circuits

These 13 QASM files are copied unmodified from the `tqe` benchmark suite of
[BAQIS-Quantum/neutral-atom-compilation](https://github.com/BAQIS-Quantum/neutral-atom-compilation)
(the ZAP compiler, MIT licensed), specifically `benchmark/tqe/`. They are
ZAP's own published-paper benchmark set (`setting/tqe.json`), used here to
reproduce their checked-in `results/tqe/log/*.json` schedule-depth numbers
as a regression check - see `tests/test_zap_reproduction.py`.

`qnn_n15.qasm` (the 14th circuit in their suite) is excluded: it uses
`cswap` and custom `gate` macros that a straight QASM-to-two-basis-types
loader doesn't need to resolve for this check, since `load_zap_transpiled_gates`
goes through Qiskit's own parser and transpiler - the exclusion here is
just about keeping the fixture set to circuits we've verified end-to-end,
not a parser limitation.
