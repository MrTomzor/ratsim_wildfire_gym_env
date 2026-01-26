# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env

from ratsim_wildfire_gym_env.env import WildfireGymEnv

worldgen_cfg = {
    "seed": 42,
    "numAgents": 1,
    "arenaWidth": 1000,
    "arenaHeight": 1000,
    "treeDensity": 0.003,
    "topology": "forest",
    "fireSpawnFrequency": 0.02,
}

sensor_cfg = {
    "lidar_num_rays": 360,
}

action_cfg = {
    "type": "diff_drive",
}

def make_env():
    return WildfireGymEnv(
        worldgen_config=worldgen_cfg,
        sensor_config=sensor_cfg,
        action_config=action_cfg,
    )

env = make_vec_env(make_env, n_envs=1)

model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="./tb",
)

model.learn(total_timesteps=1_000_000)
model.save("ppo_wildfire")
