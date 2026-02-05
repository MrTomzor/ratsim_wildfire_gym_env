from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from ratsim_wildfire_gym_env.env import WildfireGymEnv


def make_env():
    worldgen_config = {
        "seed": 66,
        "numAgents": 1,
        "startAndGoalClearingDistance": 5.0,
        "arenaWidth": 1000.0, # have to be float for proper msg conversion
        "arenaHeight": 1500.0,
        # "arenaWidth": 600,
        # "arenaHeight": 800,
        "treeDensity": 0.01,
        # "treeDensity": 0.000,
        "topology": "forest",
        # "treesSwayingFactor": 1.0,
        # "debrisTriggerzoneSpawnFrequency": 0.1,
        # "debrisGroupSizeModifier": 1.0,
        # "carRoadSpawnFrequency": 5,
        # # "carRoadSpawnFrequency": 0,
        # "carVelocityMin": 10.0,
        # "carVelocityMax": 20.0,
        # "fireSpawnFrequency": 0.02,
        # "fireGlobalSpreadModifier": 1.0,
        # "fireSmokeGenerationModifier": 1.0,
        # "fireSpreadsAcrossGround": True,
        # "staticWindXVel": 5.0,
        # "staticWindYVel": 0.0,
        # "windFluctuationModifier": 1.0,
    }

    sensor_config = {
        "lidar_num_rays": 360,
    }

    action_config = {
        "type": "diff_drive",
    }

    metaworldgen_cfg = {
        "world_generation_metaseed": 666
    }

    return WildfireGymEnv(
        worldgen_config=worldgen_config,
        sensor_config=sensor_config,
        action_config=action_config,
        metaworldgen_config=metaworldgen_cfg,
        max_steps=400,
    )


def main():
    # SB3 *requires* a vectorized env
    env = make_vec_env(make_env, n_envs=1)

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

    model.learn(total_timesteps=1_000_000)

    model.save("ppo_wildfire_trained")

    env.close()


if __name__ == "__main__":
    main()
