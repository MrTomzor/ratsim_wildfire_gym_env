from ratsim_gym_envs.forager_env_1 import *

# Test specific action sequences to verify behavior
env = ForagerEnv()
obs = env.reset(seed=42)  # Use seed for reproducible testing
# Test each action type
# actions = [0, 1, 2, 3]  # right, up, left, down

testaction = [1, 1]
for i in range(30):
    obs, reward, terminated, truncated, info = env.step(testaction)
    print("Step:", i + 1)
    if reward != 0:
        print("Non-zero reward received:", reward)




