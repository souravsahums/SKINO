import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
_res = lambda name: os.path.join(HERE, "results", name)
r = json.load(open(_res('tier2_wave.json')))
print("Wave-equation rollout diagnosis")
print(f"  {'model':>14}  test_L2     finite_steps  first_NaN_step   final_state_err")
for m in ['SKINO','SKINO-NoSymp','FNO','DeepONet','Transformer']:
    e = np.array(r[m]['energy_drift_curve'])
    s = np.array(r[m]['state_error_curve'])
    nan_mask = ~(np.isfinite(e) & np.isfinite(s))
    nan_step = int(np.argmax(nan_mask)) if nan_mask.any() else None
    print(f"  {m:>14}  {r[m]['test_relative_l2']:.3e}   {int((~nan_mask).sum())}/{len(e)}        {nan_step}            {s[-1] if np.isfinite(s[-1]) else float('nan'):.3e}")
print()
print("KdV diagnosis")
r = json.load(open(_res('tier2_kdv.json')))
for m in ['SKINO','SKINO-NoSymp','FNO','DeepONet','Transformer']:
    md = np.array(r[m]['mass_drift_curve'])
    pd = np.array(r[m]['momentum_drift_curve'])
    se = np.array(r[m]['state_error_curve'])
    print(f"  {m:>14}  test_L2={r[m]['test_relative_l2']:.3e}  state_err_final={se[-1]:.3e}  mass_drift_final={md[-1]:.3e}  momentum_drift_final={pd[-1]:.3e}")
print()
print("Tier3 porous flow diagnosis")
r = json.load(open(_res('tier3_porous_flow.json')))
for m in ['SKINO','SKINO-NoSymp','FNO','DeepONet','Transformer']:
    md = np.array(r[m]['mass_drift_curve'])
    se = np.array(r[m]['state_error_curve'])
    print(f"  {m:>14}  test_L2={r[m]['test_relative_l2']:.3e}  state_err_final={se[-1]:.3e}  mass_drift_final={md[-1]:.3e}")
