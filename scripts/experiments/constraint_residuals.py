#!/usr/bin/env python3
# Author: sous_chef, 1 July 2026
# The Idea: Before anything moves, verify the linkage is geometrically consistent at rest. If connect equality residuals are large, every
# subsequent experiment is meaningless because the car's rest pose is already wrong. This script checks constraint residuals at t=0 (after
# mj_forward only — no stepping) and then again after a short settle period to see if the solver drives residuals down or lets them grow.

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

# Threshold below which a residual is considered acceptable.
# Units are metres for connect/weld equalities, radians for joint equalities.
# Rule of thumb: should be at least 2 orders of magnitude below your
# smallest geometric feature (your linkage sites are at ~mm scale -> 1e-4 is tight).
RESIDUAL_THRESHOLD = 1e-4   # m

# How many steps to let the solver settle before the second residual check.
SETTLE_STEPS = 500


# ─── Helpers :

def get_equality_names(model):
    """Return a list of equality constraint names in model order."""
    names = []
    for i in range(model.neq):
        # model.equality returns an MjModelEquality object
        names.append(model.equality(i).name or f"eq_{i}")
    return names


def get_residuals(data, model):
    """
    Extract per-equality-constraint position residuals from data.efc_pos.

    MuJoCo packs ALL constraint rows (contact + equality + friction) into
    efc_pos. Equality constraints always come first, one row per constraint,
    so slicing [:model.neq] isolates them cleanly.
    """
    if model.neq == 0:
        return np.array([])
    return data.efc_pos[:model.neq].copy()


def print_residual_table(names, residuals, threshold):
    """Print a formatted table of constraint name -> residual -> pass/fail."""
    col_w = max(len(n) for n in names) + 2
    print(f"\n  {'Constraint':<{col_w}} {'Residual (m)':>14}  {'Status'}")
    print("  " + "-" * (col_w + 22))
    for name, res in zip(names, residuals):
        status = "PASS" if abs(res) < threshold else "FAIL"
        flag   = "  <-- !" if status == "FAIL" else ""
        print(f"  {name:<{col_w}} {res:>14.6e}  {status}{flag}")


# ─── Main :

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    if model.neq == 0:
        print("\nModel has no equality constraints — nothing to check.")
        return

    eq_names = get_equality_names(model)
    print(f"\nFound {model.neq} equality constraint(s): {eq_names}")

    # ─── CHECK 1: At rest, immediately after mj_forward (no stepping) ────────
    # This is the purest check: does the XML define a geometrically consistent
    # rest pose? If sites are misaligned in the XML, residuals will be large
    # here before the solver has had any chance to correct them.

    mujoco.mj_resetData(model, data)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)      # propagate kinematics only, do NOT step

    residuals_t0 = get_residuals(data, model)

    print("\n" + "=" * 60)
    print("CHECK 1 — Residuals at t=0 (mj_forward only, no stepping)")
    print("=" * 60)
    print_residual_table(eq_names, residuals_t0, RESIDUAL_THRESHOLD)

    # ─── CHECK 2: After SETTLE_STEPS steps with zero control ─────────────────
    # The constraint solver runs inside mj_step. This tells you whether the
    # solver converges the residuals toward zero (good) or lets them grow
    # (bad: misaligned sites, incompatible DOFs, or solver parameters too soft).

    log_time = np.zeros(SETTLE_STEPS)
    log_efc  = np.zeros((SETTLE_STEPS, model.neq))

    for i in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
        log_time[i] = data.time
        log_efc[i]  = get_residuals(data, model)

    residuals_settled = log_efc[-1]

    print(f"\n{'=' * 60}")
    print(f"CHECK 2 — Residuals after {SETTLE_STEPS} settle steps (zero control)")
    print(f"{'=' * 60}")
    print_residual_table(eq_names, residuals_settled, RESIDUAL_THRESHOLD)

    # ─── SUMMARY :

    max_t0      = np.max(np.abs(residuals_t0))
    max_settled = np.max(np.abs(residuals_settled))
    trend       = "DECREASING (solver converging)" if max_settled < max_t0 \
                  else "INCREASING (solver diverging — check solref/solimp)"

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Max residual at t=0      : {max_t0:.6e} m")
    print(f"  Max residual after settle: {max_settled:.6e} m")
    print(f"  Trend                    : {trend}")
    print(f"  Threshold                : {RESIDUAL_THRESHOLD:.1e} m")

    t0_ok      = max_t0 < RESIDUAL_THRESHOLD
    settled_ok = max_settled < RESIDUAL_THRESHOLD

    print(f"\n  CHECK 1 (geometric consistency at rest) : {'PASS' if t0_ok else 'FAIL'}")
    print(f"  CHECK 2 (solver convergence after settle): {'PASS' if settled_ok else 'FAIL'}")

    if not t0_ok:
        print("\n  ACTION: One or more sites are misaligned in the XML.")
        print("          Open the model in the MuJoCo viewer, enable")
        print("          site rendering, and nudge site positions until")
        print("          the connect constraint lines are zero-length at rest.")

    if t0_ok and not settled_ok:
        print("\n  ACTION: Rest pose is fine but solver is softening constraints.")
        print("          Try tightening solref (e.g. solref='0.005 1')")
        print("          and solimp (e.g. solimp='0.99 0.999 0.001').")

    print("=" * 60)

    # ─── PLOT :

    fig, ax = plt.subplots(figsize=(10, 5))

    for j, name in enumerate(eq_names):
        ax.plot(log_time, np.abs(log_efc[:, j]), label=name)

    ax.axhline(RESIDUAL_THRESHOLD, color='red', linestyle='--',
               linewidth=1.2, label=f"threshold = {RESIDUAL_THRESHOLD:.0e} m")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|Constraint residual| (m)")
    ax.set_title("Equality Constraint Residuals at Rest (zero control)")
    ax.legend()
    ax.grid(True)
    ax.set_yscale("log")

    plt.tight_layout()
    outfile = "results/constraint_residuals.png"
    plt.savefig(outfile, dpi=150)
    print(f"\nSaved plot to {outfile}")


if __name__ == "__main__":
    main()