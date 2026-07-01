#!/usr/bin/env python3
# Author: sous_chef, 30 Jun 2026
# The Idea: Command +δ steer, record the steady-state left and right wheel angles. Then command -δ and verify the angles mirror exactly. Any asymmetry
# points to a geometry bug in the connect constraints or cup pivot positions.

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["MPLBACKEND"] = "Agg"

# ─── Project root resolution :

def find_project_root(marker="RLAN_Supplements"):
    """Walk up from this file's directory until a folder named `marker` is found."""
    path = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.basename(path) == marker:
            return path
        parent = os.path.dirname(path)
        if parent == path:
            raise FileNotFoundError(
                f"Could not locate project root '{marker}' above {__file__}"
            )
        path = parent


PROJECT_ROOT = find_project_root()
XML_PATH = os.path.join(PROJECT_ROOT, "xml", "crickets_forward_ackermann_floored.xml")
print("Resolved XML path:", XML_PATH)
assert os.path.exists(XML_PATH), f"XML not found at {XML_PATH}"

# ─── Constants :

# Steer commands to test. Units match your actuator ctrlrange (radians). Steer-servo ctrlrange is [-0.261799, 0.261799] rad (±15 deg).
DELTA_COMMANDS  = [0.05, 0.10, 0.15, 0.20, 0.261799]  # positive deflections

# How long to hold each steer command before sampling steady state. Long enough for the linkage to settle — tune down if your damping is high.
SETTLE_DURATION = 1.0   # seconds
SETTLE_STEPS    = None  # set dynamically from model timestep

# Symmetry pass threshold: if |left + right| < this, the pair is symmetric.
# Units: degrees (converted from radians after sampling).
SYMMETRY_THRESHOLD_DEG = 0.5   # deg


# ─── Helpers :

def settle(model, data, ctrl_drive, ctrl_steer, n_steps):
    """Apply fixed controls and step for n_steps, return final state."""
    data.ctrl[drive_id] = ctrl_drive
    data.ctrl[steer_id] = ctrl_steer
    for _ in range(n_steps):
        mujoco.mj_step(model, data)


def sample_steer_angles(data, model):
    """
    Read the current steer-hinge angles for left and right front wheels. Returns (fl_deg, fr_deg) — signed, in degrees.

    If the wheels are children of the cups (current XML structure), the cup joint angles ARE the wheel steer angles. We read the cup joints.
    Fallback: if f-l-steer-hinge exists, read that instead."""
    fl_angle = None
    fr_angle = None

    # Try cup joints first (current tree: wheels are children of cups)
    cup_left_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cup-left-to-chassis")
    cup_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cup-right-to-chassis")

    if cup_left_id != -1 and cup_right_id != -1:
        fl_angle = data.qpos[model.joint(cup_left_id).qposadr[0]]
        fr_angle = data.qpos[model.joint(cup_right_id).qposadr[0]]
    else:
        # Fallback: explicit steer hinges on the wheel bodies
        fl_hinge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "f-l-steer-hinge")
        fr_hinge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "f-r-steer-hinge")
        assert fl_hinge_id != -1, "Cannot find a left steer joint — check joint names"
        assert fr_hinge_id != -1, "Cannot find a right steer joint — check joint names"
        fl_angle = data.qpos[model.joint(fl_hinge_id).qposadr[0]]
        fr_angle = data.qpos[model.joint(fr_hinge_id).qposadr[0]]

    return np.degrees(fl_angle), np.degrees(fr_angle)


# ─── Main :

def main():
    global drive_id, steer_id

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    # ─── Actuator lookups :
    drive_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive")
    steer_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "steer-servo")
    assert drive_id != -1, "Actuator 'drive' not found"
    assert steer_id != -1, "Actuator 'steer-servo' not found"

    n_settle = int(SETTLE_DURATION / model.opt.timestep)

    # ─── Result storage :
    # For each delta: store (fl_pos, fr_pos, fl_neg, fr_neg)
    results = []

    print("\n" + "=" * 70)
    print("STEERING SYMMETRY TEST")
    print(f"Settle duration : {SETTLE_DURATION} s  ({n_settle} steps at dt={model.opt.timestep})")
    print(f"Drive command   : 0.0 (wheels stationary)")
    print(f"Symmetry threshold: ±{SYMMETRY_THRESHOLD_DEG} deg")
    print("=" * 70)
    print(f"\n  {'δ (rad)':>10}  {'δ (deg)':>8}  "
          f"{'FL(+)':>8}  {'FR(+)':>8}  "
          f"{'FL(-)':>8}  {'FR(-)':>8}  "
          f"{'|FL+FR|(+)':>12}  {'|FL+FR|(-)':>12}  {'Result'}")
    print("  " + "-" * 100)

    for delta in DELTA_COMMANDS:

        # ── Positive steer command
        mujoco.mj_resetData(model, data)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        settle(model, data, ctrl_drive=0.0, ctrl_steer=+delta, n_steps=n_settle)
        fl_pos, fr_pos = sample_steer_angles(data, model)

        # ── Negative steer command
        mujoco.mj_resetData(model, data)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        settle(model, data, ctrl_drive=0.0, ctrl_steer=-delta, n_steps=n_settle)
        fl_neg, fr_neg = sample_steer_angles(data, model)

        # ── Symmetry metrics
        # For a symmetric linkage:
        #   fl_pos should equal -fl_neg  →  fl_pos + fl_neg ≈ 0
        #   fr_pos should equal -fr_neg  →  fr_pos + fr_neg ≈ 0
        asym_fl = abs(fl_pos + fl_neg)   # deviation from perfect mirror on left
        asym_fr = abs(fr_pos + fr_neg)   # deviation from perfect mirror on right
        asym_inner_outer = abs(abs(fl_pos) - abs(fr_pos))  # Ackermann differential

        passed = asym_fl < SYMMETRY_THRESHOLD_DEG and asym_fr < SYMMETRY_THRESHOLD_DEG
        status = "PASS" if passed else "FAIL <--"

        results.append({
            'delta':     delta,
            'fl_pos':    fl_pos,   'fr_pos':    fr_pos,
            'fl_neg':    fl_neg,   'fr_neg':    fr_neg,
            'asym_fl':   asym_fl,  'asym_fr':   asym_fr,
            'asym_diff': asym_inner_outer,
            'passed':    passed,
        })

        print(f"  {delta:>10.4f}  {np.degrees(delta):>8.2f}  "
              f"{fl_pos:>8.3f}  {fr_pos:>8.3f}  "
              f"{fl_neg:>8.3f}  {fr_neg:>8.3f}  "
              f"{asym_fl:>12.4f}  {asym_fr:>12.4f}  {status}")

    # ─── Summary :

    all_passed = all(r['passed'] for r in results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  δ={r['delta']:.4f} rad | "
              f"FL asym={r['asym_fl']:.4f} deg | "
              f"FR asym={r['asym_fr']:.4f} deg | "
              f"inner-outer diff={r['asym_diff']:.4f} deg | "
              f"{'PASS' if r['passed'] else 'FAIL'}")

    print()
    if all_passed:
        print("  RESULT: PASS — linkage is symmetric across all tested deflections.")
    else:
        print("  RESULT: FAIL — asymmetry detected. Likely causes:")
        print("    1. connect site positions are not mirrored in the XML")
        print("    2. cup pivot offsets differ between left and right cups")
        print("    3. one cup joint has a different damping or range limit")
        print("    4. shaft_right_end / cup_right_connection sites are misaligned")
    print("=" * 70)

    # ─── Plots ────────────────────────────────────────────────────────────────

    deltas_deg = [np.degrees(r['delta']) for r in results]
    fl_pos_vals = [r['fl_pos'] for r in results]
    fr_pos_vals = [r['fr_pos'] for r in results]
    fl_neg_vals = [r['fl_neg'] for r in results]
    fr_neg_vals = [r['fr_neg'] for r in results]
    asym_fl     = [r['asym_fl'] for r in results]
    asym_fr     = [r['asym_fr'] for r in results]
    diff_vals   = [r['asym_diff'] for r in results]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot 1: Steady-state angles for +δ and -δ
    axes[0].plot(deltas_deg, fl_pos_vals,  'o-',  color='blue',   label="FL  (+δ)")
    axes[0].plot(deltas_deg, fr_pos_vals,  's-',  color='cyan',   label="FR  (+δ)")
    axes[0].plot(deltas_deg, fl_neg_vals,  'o--', color='red',    label="FL  (-δ)")
    axes[0].plot(deltas_deg, fr_neg_vals,  's--', color='orange', label="FR  (-δ)")
    axes[0].axhline(0, color='black', linewidth=0.8, linestyle=':')
    axes[0].set_ylabel("Wheel steer angle (deg)")
    axes[0].set_title("Steady-State Steer Angles vs Command Deflection")
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Asymmetry magnitude (should be ~0 for a symmetric linkage)
    axes[1].plot(deltas_deg, asym_fl, 'o-', color='blue',   label="|FL(+δ) + FL(-δ)|")
    axes[1].plot(deltas_deg, asym_fr, 's-', color='orange', label="|FR(+δ) + FR(-δ)|")
    axes[1].axhline(SYMMETRY_THRESHOLD_DEG, color='red', linestyle='--',
                    linewidth=1.2, label=f"threshold = {SYMMETRY_THRESHOLD_DEG} deg")
    axes[1].set_ylabel("Asymmetry (deg)")
    axes[1].set_title("Mirror Asymmetry (0 = perfect symmetry)")
    axes[1].legend()
    axes[1].grid(True)

    # Plot 3: Inner vs outer wheel differential (Ackermann indicator)
    axes[2].plot(deltas_deg, diff_vals, 'D-', color='purple',
                 label="|FL angle| - |FR angle| at +δ")
    axes[2].axhline(0, color='black', linewidth=0.8, linestyle=':')
    axes[2].set_xlabel("Steer command |δ| (deg)")
    axes[2].set_ylabel("Differential angle (deg)")
    axes[2].set_title("Inner vs Outer Wheel Differential (Ackermann indicator)\n"
                      ">0 means left wheel steers sharper than right at +δ")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    outfile = "results/steering_symmetry.png"
    plt.savefig(outfile, dpi=150)
    print(f"\nSaved plot to {outfile}")


if __name__ == "__main__":
    main()