# Experiment 2A: Action Mode Reweighting

This experiment compares action-mode probabilities between the OpenWAM No-WM
and WM checkpoints. Five tasks use 10 fixed RoboTwin initial states. Each model
samples 128 trajectories per state, for 12,800 trajectories in total.

Trajectory collection records executed actions, left/right end-effector poses,
gripper values, contacts, and success. Analysis pools both methods before
standardization, PCA, and K-means. Method and success labels are not included in
the clustering features; they are joined only when producing mode probability
and mode success-rate tables.

The clustering features deliberately exclude episode length because successful
episodes terminate early. They include time-normalized relative paths, relative
grasp position and orientation, gripper-close timing, pre-contact/close approach
direction, gripper-object contact position, and arm assignment. The requested
cluster count is an upper bound; K is selected per task by silhouette score.
Diagnostics report restart stability, correlation with initial-state identity,
and contact coverage. No-WM and WM use the same policy sampling seed for each
matched `(task, state, rollout)` tuple.

## Installation

This repository expects the existing workspace layout under
`/root/data/robot`. After cloning, apply the small OpenWAM integration patch
once before collection:

```bash
cd /root/data/robot/experiment2A
./apply_openwam_patch.sh
```

The patch adds optional trajectory tracing and policy sampling seeds. Both are
activated only by Experiment 2A environment variables, so existing Experiment
3 runs keep their original behavior. Model checkpoints and generated runs are
not stored in this repository.

Run a minimal end-to-end check first:

```bash
cd /root/data/robot/experiment2A
./run_smoke.sh
```

The default allocation is model GPU 4 and simulator GPU 0. Override it without
editing JSON when those cards are occupied:

```bash
EXP2A_MODEL_GPU=6 EXP2A_SIM_GPU=1 EXP2A_PORT=8860 ./run_smoke.sh
```

When the official-checkpoint calibration occupies physical GPU 6 and port
8860, run the isolated smoke allocation (physical GPU 7 + simulator GPU 0):

```bash
./run_smoke_parallel.sh
```

Run the full protocol only after inspecting the smoke traces and clusters:

```bash
./run_full.sh
```

Collection supports `--resume`. A completed rollout requires both its trace and
result JSON. The same manifest entry is reused for all repetitions of one state,
so the simulator initial state is fixed. Each rollout receives a stable,
distinct policy sampling seed; each action-chunk generation derives its seed
from that rollout seed. This separates policy stochasticity from environment
initial-state variation and makes collection reproducible.

To resume the same shell-script run after interruption, pass its existing path:

```bash
RUN_DIR=runs/full_YYYYMMDD_HHMMSS ./run_full.sh
```

The shell scripts use the repository's `robotwin-openwam` Python directly for
analysis, so activating Conda first is not required.
