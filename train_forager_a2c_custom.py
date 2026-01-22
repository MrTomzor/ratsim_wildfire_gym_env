from ratsim_gym_envs.forager_env_1 import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
import gymnasium as gym  

# Test specific action sequences to verify behavior
env = ForagerEnv()
# Test each action type
# actions = [0, 1, 2, 3]  # right, up, left, down

# testaction = [1, 1]
# for i in range(30):
#     obs, reward, terminated, truncated, info = env.step(testaction)
#     print("Step:", i + 1)
#     if reward != 0:
#         print("Non-zero reward received:", reward)


import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# ==== Hyperparameters ====
HIDDEN_DIM = 256
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.999
TAU = 0.005
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128
ENV_NAME = 'Pendulum-v1'  # Replace with your env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Environment ====
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]

# ==== Actor Network ====
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, act_dim),
            nn.Tanh()  # Output in [-1,1]
        )

    def forward(self, obs):
        return self.net(obs) * act_limit

# ==== Critic Network ====
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))

# ==== Replay Buffer ====
class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_SIZE)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self):
        batch = random.sample(self.buffer, BATCH_SIZE)
        obs, act, rew, next_obs, done = zip(*batch)

        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        act = torch.tensor(np.array(act), dtype=torch.float32)
        rew = torch.tensor(np.array(rew), dtype=torch.float32)
        next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32)
        done = torch.tensor(np.array(done), dtype=torch.float32)
        return obs, act, rew, next_obs, done

        # return map(lambda x: torch.tensor(x, dtype=torch.float32), (obs, act, rew, next_obs, done))

    def __len__(self):
        return len(self.buffer)

# ==== DDPG Agent ====
actor = Actor(obs_dim, act_dim).to(device)
critic = Critic(obs_dim, act_dim).to(device)
target_actor = Actor(obs_dim, act_dim).to(device)
target_critic = Critic(obs_dim, act_dim).to(device)

target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

actor_opt = optim.Adam(actor.parameters(), lr=ACTOR_LR)
critic_opt = optim.Adam(critic.parameters(), lr=CRITIC_LR)

buffer = ReplayBuffer()

def soft_update(target, source):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(TAU * sp.data + (1 - TAU) * tp.data)

def train_step():
    if len(buffer) < BATCH_SIZE:
        return

    # obs, act, rew, next_obs, done = buffer.sample()
    # rew = rew.unsqueeze(1)
    # done = done.unsqueeze(1)

    obs, act, rew, next_obs, done = buffer.sample()
    rew = rew.unsqueeze(1)
    done = done.unsqueeze(1)
    obs = obs.to(device)
    act = act.to(device)
    rew = rew.to(device)
    next_obs = next_obs.to(device)
    done = done.to(device)

    # Critic loss
    with torch.no_grad():
        target_q = target_critic(next_obs, target_actor(next_obs))
        y = rew + GAMMA * (1 - done) * target_q
    critic_loss = nn.MSELoss()(critic(obs, act), y)

    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    # Actor loss
    actor_loss = -critic(obs, actor(obs)).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    # Soft update
    soft_update(target_actor, actor)
    soft_update(target_critic, critic)

# ==== Training Loop ====
EPISODES = 10000
MAX_STEPS = 10000
NOISE_SCALE = 0.1


for ep in range(EPISODES):
    ep_start = time.time()
    obs = env.reset()
    ep_reward = 0
    train_time = 0
    for step in range(MAX_STEPS):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = actor(obs_tensor).detach().cpu().numpy()[0]
        noise = NOISE_SCALE * np.random.randn(act_dim)
        action = np.clip(action + noise, -act_limit, act_limit)
        next_obs, reward, done, trunc, _ = env.step(action)

        buffer.push((obs, action, reward, next_obs, float(done)))
        obs = next_obs
        ep_reward += reward

        train_start = time.time()
        train_step()
        train_end = time.time()
        train_time += (train_end - train_start)

        # env.conn.log_connection_stats()
            

        if done or trunc:
            break

    ep_time = time.time() - ep_start 
    ep_time_no_train = time.time() - ep_start - train_time
    print(f"Episode {ep} Reward: {ep_reward:.2f} Episode Time: {ep_time:.2f}s Train Time: {train_time:.2f}s Sim: {ep_time_no_train:.2f}s")
