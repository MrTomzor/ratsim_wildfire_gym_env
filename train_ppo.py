from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from ratsim_wildfire_gym_env.env import WildfireGymEnv
from ratsim_wildfire_gym_env.curricula import *

import sys


# RECURRENT
from sb3_contrib import RecurrentPPO  # Changed from stable_baselines3 import PPO


curriculum_name = ""


def make_env():
    worldgen_config = {
        "seed": 42, # will be overridden by metaworldgen_config
        # "mainLayout" : "forest_frogger",
        "mainLayout" : "suburb",
        "numAgents": 1,
        "startAndGoalClearingDistance": 5.0,
        # "arenaWidth": 1000.0, # have to be float for proper msg conversion
        # "arenaHeight": 1500.0,
        "arenaWidth": 150.0, # have to be float for proper msg conversion
        "arenaHeight": 150.0,
        # "treeDensity": 0.01,
        "treeDensity": 0.00,
        # "treeDensity": 0.0,
        "treeOscillationEnabled" : False,
        "houseNumerosity" : 20.0,
        # "houseNumerosity" : 0.0,
        "houseDoorDefaultType" : "none",
        "rewardNumerosity" : 1.0,
        "rewardDistribution" : "houses",
        # "rewardNumerosity" : 0.02,
        # "rewardNumerosity" : 0.0,
        # "rewardNumerosity" : 0.005,
        # "rewardDistribution" : "everywhere",
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

    reward_config = {
        "hard_collision_reward" : -100,
        "reward_pickup_reward" : 20,
        "max_steps" : 500,
    }

    if curriculum_name != "":
        print(f"Using curriculum: {curriculum_name}")

    return WildfireGymEnv(
        worldgen_config=worldgen_config,
        sensor_config=sensor_config,
        action_config=action_config,
        reward_config=reward_config,
        metaworldgen_config=metaworldgen_cfg,
        curriculum_name=curriculum_name,
    )


def main():
    # Check input args for curriculum name, if provided
    if len(sys.argv) > 1:
        print("setting curriculum from command line arg")
        global curriculum_name
        curriculum_name = sys.argv[1]

    # SB3 *requires* a vectorized env
    env = make_vec_env(make_env, n_envs=1)

    # model = RecurrentPPO(
        # policy="MultiInputLstmPolicy",
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        tensorboard_log="./tb_wildfire",
    )

    max_steps = 1_000_000

    # # if curriculum is used, we want the curriculum to control the max number of steps
    # if curriculum_name != "":
    #     cur = build_curriculum(curriculum_name)
    #     max_steps = cur.get_total_length()
    #     print(f"Curriculum total length (max steps): {max_steps}")

    model.learn(total_timesteps=max_steps, callback=CustomMetricsCallback())

    model.save("models/ppo_wildfire_trained")

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
