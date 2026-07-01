#!/usr/bin/env python3
# Author: sous_chef, 30 Jun 2026
# The Idea: To check if the state space diverges over time despite equilibrium.

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["MPLBACKEND"] = "Agg"

# ─── Model Data ───────────────────────────────────────────────────────────────

def find_project_root(marker="RLAN_Supplements"):
    """
    Walk up from this file's directory until a folder named `marker` is found.
    This lets any script under RLAN_Supplements/scripts/... reference
    RLAN_Supplements/xml/... without hardcoding relative ../.. paths.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.basename(path) == marker:
            return path
        parent = os.path.dirname(path)
        if parent == path:  # reached filesystem root without finding marker
            raise FileNotFoundError(
                f"Could not locate project root '{marker}' above {__file__}"
            )
        path = parent


PROJECT_ROOT = find_project_root()
xml_path = os.path.join(PROJECT_ROOT, "xml", "crickets_forward_ackermann_floored.xml")
print("Resolved XML path:", xml_path)
assert os.path.exists(xml_path), f"XML not found at {xml_path}"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)


def get_observation():
    """Extract observation from simulation data."""
    car_position = data.qpos[:3].copy()
    car_orientation = data.qpos[3:7].copy()
    car_velocity = data.qvel[:3].copy()
    car_angvel = data.qvel[3:6].copy()

    # Compute heading angle from quaternion (yaw)
    heading_rad = np.arctan2(
        2 * (car_orientation[0] * car_orientation[3] + car_orientation[1] * car_orientation[2]),
        1 - 2 * (car_orientation[2] ** 2 + car_orientation[3] ** 2)
    )
    heading_deg = np.degrees(heading_rad)

    return {
        'position': car_position,
        'orientation': car_orientation,
        'velocity': car_velocity,
        'angvel': car_angvel,
        'heading': heading_deg
    }


def run_simulation(duration=10.0, timestep=0.01):
    """
    Run static equilibrium test: zero control input, log full state,
    and check whether anything drifts over `duration` seconds.
    """
    model.opt.timestep = timestep
    mujoco.mj_resetData(model, data)

    # Ensure zero velocities and zero control at start
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    num_steps = int(duration / timestep)

    # Pre-allocate logs
    log_time     = np.zeros(num_steps)
    log_pos      = np.zeros((num_steps, 3))
    log_vel      = np.zeros((num_steps, 3))
    log_angvel   = np.zeros((num_steps, 3))
    log_heading  = np.zeros(num_steps)
    log_efc      = np.zeros((num_steps, model.neq)) if model.neq > 0 else None

    for i in range(num_steps):
        mujoco.mj_step(model, data)

        obs = get_observation()

        log_time[i]    = data.time
        log_pos[i]     = obs['position']
        log_vel[i]     = obs['velocity']
        log_angvel[i]  = obs['angvel']
        log_heading[i] = obs['heading']

        if model.neq > 0:
            log_efc[i] = data.efc_pos[:model.neq]

        if i % 100 == 0:  # print every 100 steps to avoid console flood
            print(f"t={data.time:6.2f}s  "
                  f"pos={obs['position']}  "
                  f"vel={obs['velocity']}  "
                  f"heading={obs['heading']:7.3f} deg")

    return {
        'time': log_time,
        'pos': log_pos,
        'vel': log_vel,
        'angvel': log_angvel,
        'heading': log_heading,
        'efc': log_efc,
    }


def plot_results(results, outfile="results/static_equilibrium_output.png"):
    """Plot position, velocity, heading, and constraint residuals over time."""
    t = results['time']
    has_efc = results['efc'] is not None

    n_rows = 4 if has_efc else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=True)

    # Position drift
    axes[0].plot(t, results['pos'][:, 0], label="x")
    axes[0].plot(t, results['pos'][:, 1], label="y")
    axes[0].plot(t, results['pos'][:, 2], label="z")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("Static Equilibrium — Position Drift")
    axes[0].legend()
    axes[0].grid(True)

    # Linear velocity drift
    axes[1].plot(t, results['vel'][:, 0], label="vx")
    axes[1].plot(t, results['vel'][:, 1], label="vy")
    axes[1].plot(t, results['vel'][:, 2], label="vz")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].set_title("Linear Velocity Drift")
    axes[1].legend()
    axes[1].grid(True)

    # Heading drift
    axes[2].plot(t, results['heading'], label="heading", color="purple")
    axes[2].set_ylabel("Heading (deg)")
    axes[2].set_title("Heading Drift")
    axes[2].legend()
    axes[2].grid(True)

    # Constraint residuals (if equality constraints exist)
    if has_efc:
        axes[3].plot(t, np.max(np.abs(results['efc']), axis=1),label="max |efc_pos|", color="red")
        axes[3].set_ylabel("Constraint residual (m)")
        axes[3].set_xlabel("Time (s)")
        axes[3].set_title("Equality Constraint Residuals")
        axes[3].legend()
        axes[3].grid(True)
    else:
        axes[2].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"\nSaved plot to {outfile}")


def summarize(results):
    """Print pass/fail style summary of drift magnitudes."""
    pos_drift = np.linalg.norm(results['pos'][-1] - results['pos'][0])
    vel_final = np.linalg.norm(results['vel'][-1])
    heading_drift = abs(results['heading'][-1] - results['heading'][0])

    print("\n" + "=" * 50)
    print("STATIC EQUILIBRIUM TEST SUMMARY")
    print("=" * 50)
    print(f"  Position drift (start -> end) : {pos_drift:.6e} m")
    print(f"  Final velocity magnitude      : {vel_final:.6e} m/s")
    print(f"  Heading drift                 : {heading_drift:.6e} deg")

    if results['efc'] is not None:
        max_efc = np.max(np.abs(results['efc']))
        print(f"  Max constraint residual       : {max_efc:.6e} m")

    print("=" * 50)

    # Simple heuristic thresholds — tune to your model's scale
    ok = pos_drift < 1e-2 and vel_final < 1e-2 and heading_drift < 0.5
    print("RESULT:", "PASS — system at rest, no divergence detected" if ok
          else "FAIL — check constraints, contact, or solver settings")


if __name__ == "__main__":
    results = run_simulation(duration=10.0, timestep=0.01)
    plot_results(results)
    summarize(results)