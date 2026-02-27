from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from ratsim_wildfire_gym_env.env import WildfireGymEnv
from ratsim_wildfire_gym_env.curricula import *
from ratsim.config_blender import blend_presets, load_preset

import sys


# RECURRENT
from sb3_contrib import RecurrentPPO  # Changed from stable_baselines3 import PPO


# curriculum_name = "forest_to_houses_1"
curriculum_name = ""
is_recurrent = False

# Load base config from preset, then override for training
worldgen_config = blend_presets("world", ["default"])
worldgen_config.update({
    "seed": 42, # will be overridden by metaworldgen_config
    "tree_generation/density": 0.01,
})

agent_config = blend_presets("agents", ["sphereagent_2d_lidar"])

sensor_config = {
    # "lidar_num_rays": 360,
}

action_config = {
    "control_mode": "acceleration",
    # "control_mode": "velocity",
}

metaworldgen_cfg = {
    # "world_generation_metaseed": 666
    "world_generation_metaseed": 1
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
        agent_config=agent_config,
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
    model = None

    if is_recurrent:
        model = RecurrentPPO(
            policy="MultiInputLstmPolicy",
            env=env,
            verbose=1,
            n_steps=2048,
            batch_size=64,
            learning_rate=3e-4,
            gamma=0.99,
            tensorboard_log="./tb_wildfire",
        )

    else:
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

    model.save("models/ppo_trained")

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
