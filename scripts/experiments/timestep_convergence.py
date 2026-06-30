#!/usr/bin/env python3
# Author: sous_chef, 15 Jun 2026
# The Idea: test how the timestep affects the accuracy of the simulation, especially for fast dynamics like wheel spin. We will apply a fixed drive
# torque to the rear wheel and compare the integrated odometry (based on wheel angle) to the actual displacement of the robot. As we decrease the
# timestep, we should see the error decrease, ideally converging to zero as dt -> 0.

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["MPLBACKEND"] = "Agg"

# ─── Project root resolution (shared pattern across scripts/experiments) :

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

TIMESTEPS = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
T_SIM_DURATION = 1.0          # fixed wall-clock simulation duration (s)
R_WHEEL = 0.0135              # wheel radius (m), from XML mesh dimensions
DRIVE_CTRL = 0.5              # fixed drive motor command
STEER_CTRL = 0.0              # straight-line only — no steering


def run_single_dt(dt):
    """
    Load a fresh model/data at the given timestep, drive the rear wheel
    at a fixed control input for T_SIM_DURATION seconds, and return the
    odometry error: |x_from_wheel_angle - x_from_actual_displacement|.
    """
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # ─── Name-based lookups — never hardcode actuator/joint indices :
    drive_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive")
    steer_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "steer-servo")
    rl_drive_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r-l-drive-hinge")

    assert drive_id != -1, "Actuator 'drive' not found in model"
    assert steer_id != -1, "Actuator 'steer-servo' not found in model"
    assert rl_drive_joint_id != -1, "Joint 'r-l-drive-hinge' not found in model"

    rl_qpos_adr = model.joint(rl_drive_joint_id).qposadr[0]

    # Zero everything, set fixed control inputs
    data.qvel[:] = 0.0
    data.ctrl[drive_id] = DRIVE_CTRL
    data.ctrl[steer_id] = STEER_CTRL
    mujoco.mj_forward(model, data)

    num_steps = int(T_SIM_DURATION / dt)

    pos_start = data.qpos[0:3].copy()
    theta_start = data.qpos[rl_qpos_adr]

    for _ in range(num_steps):
        mujoco.mj_step(model, data)

    pos_end = data.qpos[0:3].copy()
    theta_end = data.qpos[rl_qpos_adr]

    # Total wheel rotation over the run (rad)
    delta_theta = theta_end - theta_start

    # Odometry-predicted displacement: arc length = r * theta
    x_odom = abs(delta_theta) * R_WHEEL

    # Actual displacement of the rover's centroid in world space
    x_actual = np.linalg.norm(pos_end - pos_start)

    error = abs(x_odom - x_actual)
    return error, x_odom, x_actual, delta_theta


def main():
    errors = []
    odom_vals = []
    actual_vals = []

    for dt in TIMESTEPS:
        error, x_odom, x_actual, delta_theta = run_single_dt(dt)
        errors.append(error)
        odom_vals.append(x_odom)
        actual_vals.append(x_actual)

        print(f"dt={dt:.4f}  "
              f"theta={delta_theta:8.4f} rad  "
              f"x_odom={x_odom:.6f} m  "
              f"x_actual={x_actual:.6f} m  "
              f"error={error:.6e} m")

    # ─── Convergence plot :
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].loglog(TIMESTEPS, errors, 'o-', color='red')
    ax[0].set_xlabel("dt (s)")
    ax[0].set_ylabel("Odometry error (m)")
    ax[0].set_title("Timestep Convergence — log-log")
    ax[0].grid(True, which="both")

    ax[1].plot(TIMESTEPS, odom_vals, 'o-', label="x_odom (wheel angle)")
    ax[1].plot(TIMESTEPS, actual_vals, 's-', label="x_actual (centroid)")
    ax[1].set_xlabel("dt (s)")
    ax[1].set_ylabel("Displacement (m)")
    ax[1].set_title("Odometry vs Actual Displacement")
    ax[1].invert_xaxis()  # finer dt to the right, easier to read trend
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    outfile = "timestep_convergence.png"
    plt.savefig(outfile, dpi=150)
    print(f"\nSaved plot to {outfile}")

    # ─── Convergence order estimate :
    # Fit log(error) = p * log(dt) + c over the finer half of the range,
    # where p is the empirical order of the integrator error.
    log_dt = np.log(TIMESTEPS)
    log_err = np.log(np.maximum(errors, 1e-12))  # avoid log(0)
    p, c = np.polyfit(log_dt, log_err, 1)

    print("\n" + "=" * 50)
    print("TIMESTEP CONVERGENCE SUMMARY")
    print("=" * 50)
    print(f"  Estimated convergence order p ≈ {p:.3f}")
    print("  (MuJoCo's default semi-implicit Euler is first-order: p ≈ 1.0 expected)")
    print(f"  Smallest dt tested: {min(TIMESTEPS)}  -> error = {errors[-1]:.6e} m")
    print("=" * 50)


if __name__ == "__main__":
    main()