"""Track 2 research program: making SKINO usable at autoregressive rollout.

Directed by Dr Gareth O'Brien. The premise (established in the seismic study)
is that BOTH FNO and SKINO fail the long-horizon 3-D rollout, so ranking them
is comparing two broken things. Track 2 instead asks a single absolute
question on *simplified, smooth* problems:

    Can SKINO be trained to roll out to a useful horizon at all?

The techniques, in the mentor's execution order, are:

    Phase 1 : (1) training-input noise injection
              (2) homogeneous / smooth problem simplification
              (4) energy-trajectory-matching penalty
              (5) extended push-forward horizon K
              (6) two-step input stencil  (u_{t-1}, u_t) -> u_{t+1}
    Phase 2 : (3) RMS-vs-horizon / breakdown-point evaluation
              (7) teacher forcing + recursive loss (curriculum)

Everything here is self-contained and imports ``skino`` directly. Run modules
as ``python -m track2.<module>`` from the repository root.
"""
