from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from ratsim_wildfire_gym_env.env import WildfireGymEnv
from ratsim_wildfire_gym_env.curricula import *

import sys
import time
from pynput import keyboard


# RECURRENT
from sb3_contrib import RecurrentPPO  # Changed from stable_baselines3 import PPO


curriculum_name = "forest_to_houses_1"

worldgen_config = {
    "seed": 42, # will be overridden by metaworldgen_config
    "mainLayout" : "suburb",
    "numAgents": 1,
    "startAndGoalClearingDistance": 5.0,
    "arenaWidth": 150.0, # have to be float for proper msg conversion
    "arenaHeight": 150.0,
    "treeDensity": 0.00,
    "treeOscillationEnabled" : False,
    "houseNumerosity" : 20.0,
    "houseDoorDefaultType" : "none",
    "rewardNumerosity" : 1.0,
    "rewardDistribution" : "houses",
}

sensor_config = {
    # "lidar_num_rays": 360,
}

action_config = {
    "control_mode": "acceleration",
    # "control_mode": "velocity",
}

metaworldgen_cfg = {
    "world_generation_metaseed": 666
}

reward_config = {}
# reward_config = {
#     "hard_collision_reward" : -100,
#     "reward_pickup_reward" : 20,
#     "max_steps" : 500,
# }


def make_env():

    return WildfireGymEnv(
        worldgen_config=worldgen_config,
        sensor_config=sensor_config,
        action_config=action_config,
        reward_config=reward_config,
        metaworldgen_config=metaworldgen_cfg,
        curriculum_name=curriculum_name,
    )


# Track currently pressed keys
pressed_keys = set()

def on_press(key):
    pressed_keys.add(key)

def on_release(key):
    pressed_keys.discard(key)

def get_action():
    """Map WASD keys to [linear, angular] action values."""
    linear = 0.0
    angular = 0.0

    if keyboard.KeyCode.from_char('w') in pressed_keys:
        linear += 1.0
    if keyboard.KeyCode.from_char('s') in pressed_keys:
        linear -= 1.0
    if keyboard.KeyCode.from_char('a') in pressed_keys:
        angular += 1.0
    if keyboard.KeyCode.from_char('d') in pressed_keys:
        angular -= 1.0

    return [linear, angular]

def main():
    if len(sys.argv) > 1:
        print("setting curriculum from command line arg")
        global curriculum_name
        curriculum_name = sys.argv[1]

    env = make_env()

    TARGET_FPS = 10
    dt = 1.0 / TARGET_FPS

    # Start listening to keyboard in non-blocking background thread
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("Use WASD to control. Press Ctrl+C to quit.")

    obs, _ = env.reset()
    terminated = False
    truncated = False

    try:
        while True:
            frame_start = time.perf_counter()

            action = get_action()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, _ = env.reset()

            # Sleep for remainder of frame to maintain 60fps
            elapsed = time.perf_counter() - frame_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        listener.stop()
        env.close()

# # #{ Metric logging callback
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class CustomMetricsCallback(BaseCallback):
    def __init__(self, log_freq=2048, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        # log every N steps (match PPO rollout length usually)
        if self.n_calls % self.log_freq == 0:

            # get values from all envs
            distances = self.training_env.env_method("get_distance_traveled")
            pickups = self.training_env.env_method("get_reward_pickups")
            longest_step_distance = self.training_env.env_method("get_longest_step_distance")

            # compute averages across vectorized envs
            avg_distance = np.mean(distances)
            avg_pickups = np.mean(pickups)

            # log to tensorboard
            self.logger.record("custom/avg_distance_traveled", avg_distance)
            self.logger.record("custom/avg_reward_pickups", avg_pickups)
            self.logger.record("custom/longest_step_distance", np.max(longest_step_distance))

        return True
# # #}

if __name__ == "__main__":
    main()
