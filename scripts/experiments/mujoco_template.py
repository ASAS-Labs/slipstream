# Author: sous_chef, 12 Jun 2026
# The Idea: Let this be the template for MuJoCo experiments.

import os
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time
os.environ["MPLBACKEND"] = "Agg"

# ─── 1. LOAD MODEL :

MODEL_PATH = "xml/crickets_forward_ackermann_floored.xml"
print(os.path.abspath(MODEL_PATH))

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)


# ─── 2. INSPECT MODEL (run once, comment out after) :

print("=" * 50)
print(f"  timestep   : {model.opt.timestep}")
print(f"  nq         : {model.nq}")        # generalised positions
print(f"  nv         : {model.nv}")        # generalised velocities
print(f"  nu         : {model.nu}")        # actuators
print(f"  neq        : {model.neq}")       # equality constraints
print(f"  nbody      : {model.nbody}")     # bodies
print("  actuators  :", [model.actuator(i).name for i in range(model.nu)])
print("  joints     :", [model.joint(i).name  for i in range(model.njnt)])
print("=" * 50)

# ─── 3. INITIAL CONDITIONS :

mujoco.mj_resetData(model, data)


# Free joint (centroid): qpos indices 0-6 → [x, y, z, qw, qx, qy, qz]
SPAWN_HEIGHT = 0.05                        # z off the ground
data.qpos[0] = 0.0                         # x
data.qpos[1] = 0.0                         # y
data.qpos[2] = SPAWN_HEIGHT                # z
data.qpos[3] = 1.0                         # qw (unit quaternion = no rotation)
data.qpos[4] = 0.0                         # qx
data.qpos[5] = 0.0                         # qy
data.qpos[6] = 0.0                         # qz

# Zero all velocities
data.qvel[:] = 0.0

# Forward kinematics: propagate qpos -> xpos/xmat for all bodies
# Always call after manually setting qpos
mujoco.mj_forward(model, data)

# ─── 4. ACTUATOR LOOKUP (by name, not hardcoded index) :

drive_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive")
steer_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "steer-servo")

# Joint lookup (for logging)
fl_spin_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "f-l-spin")
fr_spin_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "f-r-spin")
cup_left_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cup-left-to-chassis")
cup_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cup-right-to-chassis")

# ─── 5. CONTROL SETPOINTS :

data.ctrl[drive_id] = 0.0 # drive  : [-1.0,  1.0]
data.ctrl[steer_id] = 0.0 # steer  : [-0.26, 0.26] rad

# ─── 6. LOGGING SETUP :

N_STEPS   = 5000                          # simulation steps
log_time  = np.zeros(N_STEPS)
log_pos   = np.zeros((N_STEPS, 3))        # x y z of rover centroid
log_vel   = np.zeros((N_STEPS, 3))        # vx vy vz
log_efc   = np.zeros((N_STEPS, model.neq))# equality constraint residuals
log_cup_l = np.zeros(N_STEPS)             # left cup angle
log_cup_r = np.zeros(N_STEPS)             # right cup angle
log_fl    = np.zeros(N_STEPS)             # front-left spin
log_fr    = np.zeros(N_STEPS)             # front-right spin

# ─── 7. SIMULATION LOOP (headless) :

for i in range(N_STEPS):
    mujoco.mj_step(model, data)

    log_time[i]     = data.time
    log_pos[i]      = data.qpos[0:3]
    log_vel[i]      = data.qvel[0:3]
    log_cup_l[i]    = data.qpos[model.joint(cup_left_id).qposadr]
    log_cup_r[i]    = data.qpos[model.joint(cup_right_id).qposadr]
    log_fl[i]       = data.qpos[model.joint(fl_spin_id).qposadr]
    log_fr[i]       = data.qpos[model.joint(fr_spin_id).qposadr]

    # Constraint residuals: one value per equality constraint
    if model.neq > 0:
        log_efc[i] = data.efc_pos[:model.neq]

# ─── 8. WITH VIEWER (interactive, swap in place of loop above) :

def run_with_viewer():

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.azimuth   = 90
        v.cam.elevation = -20
        v.cam.distance  = 0.8
        v.cam.lookat[:] = [0.0, 0.0, 0.05]

        step = 0
        while v.is_running() and step < N_STEPS:
            step_start = time.time()
            mujoco.mj_step(model, data)

            # Set controls mid-sim if needed
            # data.ctrl[drive_id] = some_policy(data)

            v.sync()
            step += 1
            time.sleep(model.opt.timestep)  # real-time pacing

# Uncomment to run with viewer instead:
run_with_viewer()

# ─── 9. QUICK PLOTS :

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(log_time, log_pos[:, 0], label="x")
axes[0].plot(log_time, log_pos[:, 1], label="y")
axes[0].plot(log_time, log_pos[:, 2], label="z")
axes[0].set_ylabel("Position (m)")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(log_time, np.degrees(log_cup_l), label="cup left")
axes[1].plot(log_time, np.degrees(log_cup_r), label="cup right")
axes[1].set_ylabel("Steer angle (deg)")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(log_time, np.max(np.abs(log_efc), axis=1), label="max |efc|", color="red")
axes[2].set_ylabel("Constraint residual (m)")
axes[2].set_xlabel("Time (s)")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("experiment_output.png", dpi=150)
plt.show()