# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gymnasium reinforcement learning environment for training agents in the Ratsim wildfire simulator. Connects to a Unity-based simulator via `RoslikeUnityConnector` (from the `ratsim` package) using topic-based message passing. Trains with Stable Baselines3 (PPO, RecurrentPPO, Dreamer).

## Commands

```bash
# Install package (editable)
pip install -e .

# Train (PPO is the main algorithm)
python train_ppo.py                          # default curriculum
python train_ppo.py forest_to_houses_1       # named curriculum

# Evaluate a trained model
python test_model.py <model_path> [worldconfig] [num_episodes]
python test_model.py models/ppo_trained.zip forest_foraging_easy_1 20

# Manual keyboard control (WASD) for debugging
python test_manual_control.py [curriculum_name]

# Random agent test
python test_random_agent.py

# TensorBoard
tensorboard --logdir ./tb_wildfire
```

No test suite exists; testing is done via the `test_*.py` evaluation/interaction scripts above.

## Architecture

### Core Environment (`ratsim_wildfire_gym_env/env.py`)

`WildfireGymEnv(gym.Env)` — the main class. Implements standard Gymnasium API (`reset()`, `step()`, `close()`).

**Configuration**: config dicts control behavior:
- `worldgen_config`: Arena size, tree density, house count, reward distribution — sent to Unity each `reset()`
- `agent_config`: Agent prefab, sensors, actuators — loaded via `blend_presets("agents", ...)`, sent to Unity once during `__init__()` on `/sim_control/agent_config`
- `sensor_config`: Lidar parameters (ray count, range, semantics)
- `action_config`: Control mode (`"velocity"` or `"acceleration"`)
- `task_config`: Passed to `TaskTracker` (episode_max_steps, foraging_settings, collision_settings, termination_settings, optional volumetric_exploration_settings). Replaces the old `reward_config`.

**Volumetric exploration reward**: when `task_config` has a `volumetric_exploration_settings` block, TaskTracker builds a 2D occupancy grid from ground-truth pose + lidar scans and rewards newly-known cell area × `reward_per_m2` each step. Uses the agent's `/<name_prefix>/gt_pose` topic (Unity force-enables `AbsolutePose2DSensor` for this — it's infrastructure, not an observation). Keys: `reward_per_m2`, `grid_resolution`, `visualize`, `debug`, `debug_every`. Example preset: `task_presets/volumetric_exploration_1000_collision_penalty.yaml`.

**Optional logging kwargs** on `WildfireGymEnv.__init__`:
- `episode_log_path`: Path to a JSONL file. If set, the env appends one JSON line per completed episode (on `terminated` or `truncated`) via `_log_episode_jsonl()`. Training scripts point this at `results/<run>/train_episodes.jsonl`.
- `run_metadata`: Dict merged into every line (e.g., `{"method": "ppo", "rundef": "...", "seed": 1, "stage_idx": 0}`). Kept per-line deliberately so each JSONL is self-describing for downstream tools.

**Multi-instance / vectorized training kwarg**:
- `unity_port`: TCP port the connector attaches to (default 9000). For vectorized training, allocate ports via `ratsim.unity_launcher.allocate_unity_instances(n_envs)` and pass each env factory its own port. The `unity_launcher` handles two tiers: with `RATSIM_UNITY_BIN` set, it auto-spawns Unity instances on the 9100+ range for `n_envs>1`; without the env var, only `n_envs=1` works (reuses the manually-launched Unity on port 9000). See `ratsim/CLAUDE.md` for the full launcher contract.

`episode_idx` is **cumulative across stages**: on construction, the env counts existing lines in `episode_log_path` and uses that as its offset, so restarting the env for a new stage in the same run dir continues the numbering. `get_num_episodes()` returns the in-memory counter.

**Observation space** (Dict):
- `lidar`: Normalized depth (+ optional semantic descriptors), range 0–1
- `gps`: Agent position relative to start, normalized by factor 300.0
- `compass`: Heading angle normalized to [-1, 1]
- `goal`: Relative vector to goal in agent's local frame

**Action space** (Box, 2D): `[forward_vel/accel, angular_vel/accel]`

**ROS-like topics** (via `RoslikeUnityConnector`):
- Subscribed: `/lidar2d`, `/rat1_pose`, `/rat1_pose_from_start`, `/wildfire_goal_position`, `/collisions`, `/reward_pickup`, `/odom`
- Published: `/cmd_vel` or `/cmd_accel`, `/sim_control/world_config`, `/sim_control/agent_config`, `/sim_control/reset_episode`, `/sim_control/scene_select`

### Curriculum System (`ratsim_wildfire_gym_env/curricula.py`)

`Curriculum` class manages multi-chapter training progression. Each chapter has its own env config and episode count. `get_named_envconfig(name)` returns a 4-tuple of config dicts. `build_curriculum(name)` creates a `Curriculum` object.

### Training (`train_ppo.py`)

Uses SB3's `PPO` or `RecurrentPPO`. Custom `CustomMetricsCallback` logs distance traveled, reward pickups, and longest step distance to TensorBoard every 2048 steps. Models saved to `models/`.

### Legacy

`forager_env_1.py` is an older environment variant (simpler, no curricula, no goal vector). `train_forager_a2c_custom.py` and `train_dreamer.py` are experimental.

## Key Patterns

- Termination is driven by `TaskTracker` (collision / zero_health / zero_battery / all_rewards_collected → `terminated=True`); `truncated=True` on reaching `episode_max_steps`
- On episode end, if `episode_log_path` is set, one JSONL line is appended with the `run_metadata` plus TaskTracker's metrics (`total_score`, `objects_found`, `collisions`, `termination_reason`, `distance_traveled`, `steps`, `wall_time_s`, `episode_idx`)
- Agent config is sent once during `__init__()` via `/sim_control/agent_config` (Unity's `AgentLoader` spawns the agent each episode based on this)
- World generation happens in `reset()` via publishing world config JSON to `/sim_control/world_config` then triggering `/sim_control/reset_episode`
- `metaworldgen_config` controls deterministic seed generation for reproducible procedural environments
- Collision detection uses a buffered approach (collision velocity tracked via `/collisions` topic)
