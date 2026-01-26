import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ratsim.roslike_unity_connector.connector import RoslikeUnityConnector
from ratsim.roslike_unity_connector.message_definitions import (
    WildfireWorldGenMessage,
    StringMessage,
    Twist2DMessage,
    Lidar2DMessage,
)

class WildfireGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        worldgen_config: dict,
        sensor_config: dict,
        action_config: dict,
        max_steps: int = 1000,
    ):
        super().__init__()

        self.worldgen_config = worldgen_config
        self.sensor_config = sensor_config
        self.action_config = action_config
        self.max_steps = max_steps
        self.step_count = 0

        # --- Connect to Ratsim ---
        self.conn = RoslikeUnityConnector()
        self.conn.connect()

        # --- Select scene ---
        self._select_scene("Wildfire")
        print("Selected Wildfire scene in Ratsim.")



        # --- Set up action params, read from configs ---
        self.max_forward_velocity = self.action_config.get("max_forward_velocity", 5.0)
        self.max_angular_velocity = self.action_config.get("max_angular_velocity", 1.0)
        self.twist_msg_out_topic = "/cmd_vel"
        print(f"Action config: max_forward_velocity={self.max_forward_velocity}, max_angular_velocity={self.max_angular_velocity}")

        # --- Action space ---
        # Example: differential drive
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

        # --- Set sensing params ---
        # TODO - implement if needed
        self.num_rays = 19
        self.lidar_msg_in_topic = "/lidar2d"
        self.goal_pose_msg_in_topic = "/wildfire_goal_position"
        self.agent_pose_msg_in_topic = "/rat1_pose"

        # --- Observation space ---
        # Example: lidar + goal vector
        lidar_dim = self.num_rays
        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(0.0, 100.0, shape=(lidar_dim,), dtype=np.float32),
            "goal": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        })

    # ---------------------------
    # Gym API
    # ---------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # --- World generation ---
        worldgen_msg = self._build_worldgen_msg(options)
        self.conn.publish(worldgen_msg, "/wildfire_worldgen_input")
        self.conn.send_messages_and_step(enable_physics_step=False)

        # --- Read initial observations ---
        obs_msgs = self.conn.read_messages_from_unity()
        print("RECEIVED MSGS:")
        print(obs_msgs)
        obs = self._parse_observation(obs_msgs)

        return obs, {}

    def step(self, action):
        self.step_count += 1

        # --- Send action ---
        action_msg = self._build_action_msg(action)
        self.conn.publish(action_msg, "/cmd_vel")

        self.conn.send_messages_and_step(enable_physics_step=True)
        msgs = self.conn.read_messages_from_unity()
        print("RECEIVED MSGS:")
        print(msgs)

        obs = self._parse_observation(msgs)
        reward = self._compute_reward(msgs)
        terminated = self._check_terminated(msgs)
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def close(self):
        self.conn.close()

    # ---------------------------
    # Helpers
    # ---------------------------

    def _select_scene(self, scene_name: str):
        msg = StringMessage(data=scene_name)
        self.conn.publish(msg, "/sim_control/scene_select")
        self.conn.send_messages_and_step(enable_physics_step=False)
        self.conn.read_messages_from_unity()

    def _build_worldgen_msg(self, options):
        cfg = dict(self.worldgen_config)
        if options is not None:
            cfg.update(options)

        msg = WildfireWorldGenMessage()
        for k, v in cfg.items():
            setattr(msg, k, v)
        return msg

    def _build_action_msg(self, action):
        # Replace with your real message
        msg = Twist2DMessage()
        msg.forward = float(action[0]) * self.max_forward_velocity
        msg.left = 0
        msg.radiansCounterClockwise = float(action[1]) * self.max_angular_velocity
        return msg

    def _extract_relative_goal_vector(self, msgs):
        res = np.zeros(2, dtype=np.float32)

        if not self.goal_pose_msg_in_topic in msgs.keys() or not self.agent_pose_msg_in_topic in msgs.keys():
            print("Warning: goal or agent pose message not found in msgs.")
            print("Which topics are available:", list(msgs.keys()))
            return res

        goal_msg = msgs[self.goal_pose_msg_in_topic][0]
        agent_msg = msgs[self.agent_pose_msg_in_topic][0]

        gx = goal_msg.forward
        gy = goal_msg.left

        ax = agent_msg.forward
        ay = agent_msg.left
        a_theta = agent_msg.radiansCounterClockwise

        # Compute relative vector, taking into account agent orientation
        dx = gx - ax
        dy = gy - ay
        cos_theta = np.cos(-a_theta)
        sin_theta = np.sin(-a_theta)
        rel_x = cos_theta * dx - sin_theta * dy
        rel_y = sin_theta * dx + cos_theta * dy
        res[0] = rel_x
        res[1] = rel_y
        return res



    def _parse_observation(self, msgs):
        # This is where your sensor parsing lives
        lidar = self._extract_lidar(msgs)
        goal = self._extract_relative_goal_vector(msgs)

        print("Relative goal vector:", goal)

        return {
            "lidar": lidar,
            "goal": goal,
        }

    def _compute_reward(self, msgs):
        # Placeholder
        return -0.01

    def _check_terminated(self, msgs):
        return False

    # --- Sensor helpers ---
    def _extract_lidar(self, msgs):
        # TODO: parse lidar topic, just the distances
        if not self.lidar_msg_in_topic in msgs.keys():
            print("Warning: lidar message not found in msgs.")
            print("Which topics are available:", list(msgs.keys()))
            return np.zeros(self.observation_space["lidar"].shape, dtype=np.float32)
        lidar_msg = msgs[self.lidar_msg_in_topic][0]

        # Lidar has -1 for out of range, convert to max range
        dists = np.array(lidar_msg.ranges, dtype=np.float32)
        dists[dists < 0] = lidar_msg.maxRange
        return dists

    def _extract_goal(self, msgs):
        return np.zeros(2, dtype=np.float32)
