from ratsim_gym_envs.forager_env_1 import *
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments

# env = ForagerEnv()
# vec_env = make_vec_env("CartPole-v1", n_envs=4)
vec_env = make_vec_env(ForagerEnv, n_envs=1)

# model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")
model = PPO.load("ppo_forager")
model.set_env(vec_env)

# LEARN
num_epochs = 10
epoch_steps = 10000
for i in range(num_epochs):
    print("EPOCH " + str(i))
    model.learn(total_timesteps=10000)
    model.save("ppo_forager2")
    print("TRAINING COMPLETE, SAVED for epoch" + str(i))

del model # remove to demonstrate saving and loading


obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    # vec_env.render("human")

# print(f"Episode {ep} Reward: {ep_reward:.2f} Episode Time: {ep_time:.2f}s Train Time: {train_time:.2f}s Sim: {ep_time_no_train:.2f}s")
