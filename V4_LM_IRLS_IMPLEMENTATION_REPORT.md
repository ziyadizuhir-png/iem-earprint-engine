# vNext4 LM + Huber IRLS Implementation Report

## Solver flow

```text
Original 711 Pudding FR
        +
Selected final Robust Target
        |
        v
Scalar level alignment
        |
        v
Residual feature analysis
        |
        v
Evidence-driven candidate pool
        |
        v
Active set: 3 -> 5 -> 7 -> <=10
        |
        v
Joint LM: log(Fc), Gain, log(Q)
        |
        v
Huber IRLS robust refinement
        |
        v
Bounded multi-metric acceptance
        |
        v
Prune / redundancy cleanup
        |
        v
Dense exact RBJ refinement
        |
        v
Quantize + exact local rescue
        |
        v
12–20 kHz safety guard
        |
        v
Perturbation stability test
        |
        v
Final PEQ export
```

## LM mathematics

For active parameter vector

`theta = [log(Fc1), G1, log(Q1), ..., log(FcN), GN, log(QN)]`

the exact corrected response is evaluated from cascaded RBJ peaking biquads. The numerical Jacobian is built from central finite differences of each individual exact filter response.

Each LM step solves:

`(J^T W J + lambda * D) delta = -J^T W e`

where `D` is the damped diagonal and `W` is the current IRLS weight matrix. Failed/non-improving steps are rolled back and increase `lambda`; accepted steps reduce `lambda`.

## Huber IRLS

For residual `e`, after the small dead zone:

- `w = 1` inside the Huber delta.
- `w = delta / |e|` outside the Huber delta.

The robust state is accepted only when its Huber fitting cost actually improves and RMSE/P95/max plus the secondary objective remain bounded. Robust fitting therefore cannot replace or redefine the Robust Target.

## Band allocation

The 10-band limit is part of the solve. The solver grows the active set and validates marginal improvement at each stage. It does not generate more than the device budget and slice afterward.

## Stability and safety

- Every exported biquad is pole-checked.
- Final solutions receive deterministic Fc/Gain/Q perturbation tests.
- Exported quantized values are exactly re-simulated.
- 12–20 kHz is never a personal fitting target; it is a guard region only.
- If near-12 kHz positive filters violate HF safety, a deterministic exact-response rescue searches coherent frequency/Q/gain transforms and selects the best safety-passing fit.

## Browser implementation note

The full production solver is integrated directly in `app/pudding_peq.js`. The previous unused solver-foundation ES modules were removed so there is only one production implementation and no misleading duplicate solver path.
