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

**Configuration**: Five config dicts control behavior:
- `worldgen_config`: Arena size, tree density, house count, reward distribution — sent to Unity each `reset()`
- `agent_config`: Agent prefab, sensors, actuators — loaded via `blend_presets("agents", ...)`, sent to Unity once during `__init__()` on `/sim_control/agent_config`
- `sensor_config`: Lidar parameters (ray count, range, semantics)
- `action_config`: Control mode (`"velocity"` or `"acceleration"`)
- `reward_config`: Reward values, max steps per episode

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

- Episode terminates on collision (terminated=True) or after max_steps (truncated=True)
- Agent config is sent once during `__init__()` via `/sim_control/agent_config` (Unity's `AgentLoader` spawns the agent each episode based on this)
- World generation happens in `reset()` via publishing world config JSON to `/sim_control/world_config` then triggering `/sim_control/reset_episode`
- `metaworldgen_config` controls deterministic seed generation for reproducible procedural environments
- Collision detection uses a buffered approach (collision velocity tracked via `/collisions` topic)
