from ratsim_wildfire_gym_env.env import WildfireGymEnv
import time
import numpy as np

def main():
    # ---------------------------
    # Configuration
    # ---------------------------
    worldgen_config = {
        "seed": 42,
        "numAgents": 1,
        "startAndGoalClearingDistance": 5.0,
        "arenaWidth": 1000,
        "arenaHeight": 1000,
        "treeDensity": 0.003,
        "topology": "forest",
        "treesSwayingFactor": 1.0,
        "debrisTriggerzoneSpawnFrequency": 0.1,
        "debrisGroupSizeModifier": 1.0,
        "carRoadSpawnFrequency": 0.05,
        "carVelocityMin": 10.0,
        "carVelocityMax": 20.0,
        "fireSpawnFrequency": 0.02,
        "fireGlobalSpreadModifier": 1.0,
        "fireSmokeGenerationModifier": 1.0,
        "fireSpreadsAcrossGround": True,
        "staticWindXVel": 5.0,
        "staticWindYVel": 0.0,
        "windFluctuationModifier": 1.0,
    }

    sensor_config = {
        "lidar_num_rays": 360,
    }

    action_config = {
        "type": "diff_drive",
    }

    # ---------------------------
    # Create environment
    # ---------------------------
    env = WildfireGymEnv(
        worldgen_config=worldgen_config,
        sensor_config=sensor_config,
        action_config=action_config,
        max_steps=200,
    )

    # ---------------------------
    # Run random policy
    # ---------------------------
    obs, info = env.reset()
    print("Initial observation keys:", obs.keys())

    total_reward = 0.0

    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        print(
            f"step={step:03d} "
            f"reward={reward:.3f} "
            f"terminated={terminated} "
            f"truncated={truncated}"
        )

        # Slow down so you can observe Unity
        # time.sleep(0.05)

        if terminated or truncated:
            print("Episode finished")
            break

    print("Total reward:", total_reward)

    env.close()


if __name__ == "__main__":
    main()
