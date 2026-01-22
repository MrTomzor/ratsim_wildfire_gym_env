import gymnasium as gym
from roslike_unity_connector.connector import *
import numpy as np

class ForagerEnv(gym.Env):
    def __init__(self, maxvel = 50, maxangvel = 20.0, max_episode_seconds = 7, reward_scale = 10.0, verbose=True):
        super(ForagerEnv, self).__init__()

        # Init ROSlike conn
        self.conn = RoslikeUnityConnector(verbose=False)
        self.conn.connect()

        self.maxvel = maxvel
        self.maxangvel = maxangvel
        self.reward_scale = reward_scale
        self.max_episode_steps = (int)(max_episode_seconds / 0.02)
        self.steps_taken = 0
        print(f"ForagerEnv: max_episode_steps set to {self.max_episode_steps}")

        # Do one step to get first lidar msg and assign dimensions
        print("ForagerEnv: Initializing environment and getting first lidar message...")
        self.conn.send_messages_and_step()
        self.conn.read_messages_from_unity()
        self.latest_lidar_msg = self.conn.get_received_messages("/lidar2d")[0]
        print("ForagerEnv: First lidar message received.")

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))  # normalized forward velocity and angular velocity
        num_obsvs = len(self.latest_lidar_msg.ranges) + len(self.latest_lidar_msg.descriptors)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(num_obsvs,))

        self.observation = self.observation_space.sample()  # Initialize with a random observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.steps_taken = 0

        # Reset the environment in Unity
        self.conn.publish(StringMessage(""), "/respawn_rat")
        self.conn.send_messages_and_step()
        self.conn.read_messages_from_unity()
        self.latest_lidar_msg = self.conn.get_received_messages("/lidar2d")[0]
        observation = self.lidar_msg_to_observation(self.latest_lidar_msg)

        # Reset the environment to an initial state
        # return self.observation_space.sample()  # Return a random observation for now
        return observation, None  # Return a random observation for now

    def step(self, action):
        # Apply the action and return the new state, reward, done flag, and info

        # Convert the action to a Twist2DMessage
        twist_msg = self.model_output_to_twist_msg(action)

        # Handle comms with Unity
        self.conn.publish(twist_msg, "/cmd_vel")
        self.conn.send_messages_and_step()
        self.conn.read_messages_from_unity()
        self.latest_lidar_msg = self.conn.get_received_messages("/lidar2d")[0]
        # self.conn.log_connection_stats() 

        # Convert the latest lidar message to an observation
        observation = self.lidar_msg_to_observation(self.latest_lidar_msg)

        # Count all collected rewards
        reward = 0  # Placeholder reward
        pickup_msgs = self.conn.get_received_messages("/reward_basic")
        for msg in pickup_msgs:
            if isinstance(msg, Int32Message):
                reward += msg.data * self.reward_scale
            else:
                print(f"Unexpected message type: {type(msg)} in rewards topic")

        self.steps_taken += 1
        # print(f"Step {self.steps_taken}: Received reward {reward}, total steps {self.steps_taken}")
        done = self.steps_taken >= self.max_episode_steps
        info = {}  # Additional info
        truncated = False

        return observation, reward, done, truncated, info

    def model_output_to_twist_msg(self, model_output):
        norm_forward_vel = model_output[0]
        if norm_forward_vel < 0:
            norm_forward_vel *= 0.2;
        # norm_forward_vel = (1 + model_output[0]) / 2
        norm_angular_vel = model_output[1]

        forward_vel = norm_forward_vel * self.maxvel
        angular_vel = norm_angular_vel * self.maxangvel

        return Twist2DMessage(forward_vel, 0, angular_vel)

    def lidar_msg_to_observation(self, lidar_msg : Lidar2DMessage):
        # Convert the lidar message to an observation vector
        ranges = np.array(lidar_msg.ranges)
        descriptors = np.array(lidar_msg.descriptors)
        maxrange = lidar_msg.maxRange

        # Normalize ranges
        ranges[ranges > maxrange] = maxrange
        ranges[ranges < 0] = maxrange  # Handle invalid ranges
        ranges = ranges / maxrange

        # Normalize descriptors
        desc_max_value = np.max(np.abs(descriptors))
        descriptors = descriptors / desc_max_value if desc_max_value != 0 else descriptors

        # Combine ranges and descriptors into a single observation vector
        observation = np.concatenate((ranges, descriptors))
        return observation
