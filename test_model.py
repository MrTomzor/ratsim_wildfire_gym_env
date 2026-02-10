from ratsim_wildfire_gym_env.curricula import get_named_worldconfig
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from ratsim_wildfire_gym_env.env import WildfireGymEnv
from ratsim_wildfire_gym_env.curricula import *
import sys
import numpy as np

worldconfig_name = ""

def make_env():
    worldgen_config = {
        "seed": 42,
        "mainLayout": "suburb",
        "numAgents": 1,
        "startAndGoalClearingDistance": 5.0,
        "arenaWidth": 300.0,
        "arenaHeight": 300.0,
        "treeDensity": 0.01,
        "treeOscillationEnabled": False,
        "houseNumerosity": 0.0,
        "houseDoorDefaultType": "none",
        "rewardNumerosity": 0.005,
        "rewardDistribution": "everywhere",
    }
    sensor_config = {}
    action_config = {
        "control_mode": "velocity",
    }
    metaworldgen_cfg = {
        "world_generation_metaseed": 666
    }
    reward_config = {
        "hard_collision_reward": -100,
        "reward_pickup_reward": 20,
    }
    
    print(f"Testing with worldname: {worldconfig_name}")
    if worldconfig_name != "":
        worldgen_config, sensor_config, action_config, reward_config = get_named_worldconfig(worldconfig_name)
    
    return WildfireGymEnv(
        worldgen_config=worldgen_config,
        sensor_config=sensor_config,
        action_config=action_config,
        reward_config=reward_config,
        metaworldgen_config=metaworldgen_cfg,
        # max_steps=800,
    )

def test_model(model_path, num_episodes=10, render=False):
    """
    Test a trained model for a given number of episodes.
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to test
        render: Whether to render the environment (if supported)
    """
    # Load the trained model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create environment
    env = make_vec_env(make_env, n_envs=1)
    
    # Statistics tracking
    episode_rewards = []
    episode_lengths = []
    total_pickups = []
    total_distances = []
    
    print(f"\nTesting for {num_episodes} episodes...")
    print("-" * 50)
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Use the trained model to predict action
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            episode_length += 1
            
            if render:
                env.render()
        
        # Get custom metrics from the environment
        pickups = env.env_method("get_reward_pickups")[0]
        distance = env.env_method("get_distance_traveled")[0]
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        total_pickups.append(pickups)
        total_distances.append(distance)
        
        print(f"Episode {episode + 1}/{num_episodes}:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length}")
        print(f"  Pickups: {pickups}")
        print(f"  Distance: {distance:.2f}")
        print()
    
    # Print summary statistics
    print("=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Average Pickups: {np.mean(total_pickups):.2f} ± {np.std(total_pickups):.2f}")
    print(f"Average Distance: {np.mean(total_distances):.2f} ± {np.std(total_distances):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print("=" * 50)
    
    env.close()
    
    return {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "pickups": total_pickups,
        "distances": total_distances,
    }

def main():
    global worldconfig_name
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test.py <model_path> [worldconfig_name] [num_episodes]")
        print("Example: python test.py models/ppo_wildfire_trained.zip forest_foraging_easy_1 20")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        worldconfig_name = sys.argv[2]
        print(f"Using worldname: {worldconfig_name}")
    
    num_episodes = 10  # default
    if len(sys.argv) > 3:
        num_episodes = int(sys.argv[3])
    
    # Run the test
    results = test_model(model_path, num_episodes=num_episodes, render=False)

if __name__ == "__main__":
    main()
