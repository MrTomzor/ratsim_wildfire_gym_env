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
        metaworldgen_config: dict,
        max_steps: int = 1000,
    ):
        super().__init__()

        self.worldgen_config = worldgen_config
        self.sensor_config = sensor_config
        self.action_config = action_config
        self.max_steps = max_steps
        self.step_count = 0

        # TODO - metaparams 
        # self.goal_observation_format = "normalized_deltavec" # or deltavec, or heading
        # self.goal_observation_format = "deltavec" # or deltavec, or heading
        self.goal_observation_format = "heading" # or deltavec, or heading
        self.lidar_observation_format = "depth_only"
        self.lidar_enabled = True

        # --- Connect to Ratsim ---
        self.conn = RoslikeUnityConnector(verbose=False)
        self.conn.connect()

        # --- Select scene ---
        self._select_scene("Wildfire")
        print("Selected Wildfire scene in Ratsim.")


        # --- Set up action params, read from configs ---
        self.max_forward_velocity = self.action_config.get("max_forward_velocity", 20.0)
        self.max_angular_velocity = self.action_config.get("max_angular_velocity", 1.0)
        self.max_forward_acceleration = self.action_config.get("max_forward_acceleration", 10.0)
        self.max_angular_velocity = self.action_config.get("max_forward_acceleration", 10.0)
        self.vel_twist_msg_out_topic = "/cmd_vel"
        self.accel_twist_msg_out_topic = "/cmd_accel"
        # self.control_mode = self.action_config.get("control_mode", "velocity") 
        self.control_mode = self.action_config.get("control_mode", "acceleration") 
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
        self.num_rays = 25
        self.lidar_max_range = 60
        self.lidar_msg_in_topic = "/lidar2d"
        self.goal_pose_msg_in_topic = "/wildfire_goal_position"
        self.agent_pose_msg_in_topic = "/rat1_pose"

        # --- Observation space ---
        # Example: lidar + goal vector
        obsvdict = {}
        if self.lidar_enabled:
            if self.lidar_observation_format == "depth_only":
                # obsvdict["lidar"] = spaces.Box(0.0, 100.0, shape=(self.num_rays,), dtype=np.float32)
                obsvdict["lidar"] = spaces.Box(0.0, 1.0, shape=(self.num_rays,), dtype=np.float32) # normalized
            else:
                print("Warning: unknown lidar observation format")
                return 1

        if self.goal_observation_format == "normalized_deltavec":
            obsvdict["goal"] = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        elif self.goal_observation_format == "deltavec":
            obsvdict["goal"] = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        elif self.goal_observation_format == "heading":
            # obsvdict["goal"] = spaces.Box(-np.pi, np.pi, shape=(1,), dtype=np.float32)
            obsvdict["goal"] = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        else:
            print("Warning: unknown goal observation format, defaulting to normalized_deltavec")
            return 1

        lidar_dim = self.num_rays
        self.observation_space = spaces.Dict(obsvdict)

        # print dimensionality of observation space
        print("Observation space:", self.observation_space)

        # Random generaiton preparation
        # Create objects necessary for generating a world seed using the metaseed on every reset
        self.metaworldgen_config = metaworldgen_config
        self.world_seed_generator = None
        if self.metaworldgen_config is not None and "world_generation_metaseed" in self.metaworldgen_config:
            seed_generation_seed = metaworldgen_config["world_generation_metaseed"]
            self.world_seed_generator = np.random.default_rng(seed_generation_seed)

        # Reward init
        self.collision_reward = -10

    # ---------------------------
    # Gym API
    # ---------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # --- World generation ---
        worldgen_msg = self._build_worldgen_msg(options)

        # Change seed if using metaworldgen
        if self.world_seed_generator is not None:
            new_world_seed = int(self.world_seed_generator.integers(0, 2**31 - 1))
            print(f"Generated new world seed: {new_world_seed}")
            worldgen_msg.seed = new_world_seed

        self.conn.publish(worldgen_msg, "/wildfire_worldgen_input")
        self.conn.send_messages_and_step(enable_physics_step=False)

        # --- Read initial observations ---
        obs_msgs = self.conn.read_messages_from_unity()
        obs = self._parse_observation(obs_msgs)

        # -- Reset goal nearing tracking ---

        # Write initial best goal distance if was initialized before
        if hasattr(self, 'best_goal_distance'):
            print("best goal distance:" + str(self.best_goal_distance))
        goal_vector = self._extract_relative_goal_vector(obs_msgs)
        self.best_goal_distance = np.linalg.norm(goal_vector)

        return obs, {}

    def step(self, action):
        self.step_count += 1

        # --- Send action ---
        action_msg = self._build_action_msg(action)
        if self.control_mode == "acceleration":
            self.conn.publish(action_msg, self.accel_twist_msg_out_topic)
        else:
            self.conn.publish(action_msg, self.vel_twist_msg_out_topic)

        self.conn.send_messages_and_step(enable_physics_step=True)
        msgs = self.conn.read_messages_from_unity()

        obs = self._parse_observation(msgs)
        reward = self._compute_reward(msgs)
        terminated = self._check_terminated(msgs)
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def close(self):
        # self.conn.close()
        print("Closing WildfireGymEnv.")
        return

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
        if self.control_mode == "acceleration":
            # Acceleration control
            msg.forward = float(action[0]) * self.max_forward_acceleration
            msg.left = 0
            msg.radiansCounterClockwise = float(action[1]) * self.max_angular_velocity
        else:
            # Velocity control
            msg.forward = float(action[0]) * self.max_forward_velocity
            msg.left = 0
            msg.radiansCounterClockwise = float(action[1]) * self.max_angular_velocity
        return msg

    def _get_collision_vel_if_collided(self, msgs):
        collision_topic = "/collisions"
        if not collision_topic in msgs.keys():
            return None
        collision_msg = msgs[collision_topic][0]
        return collision_msg.data


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

        # Normalize goal vec if needed
        if self.goal_observation_format == "normalized_deltavec":
            norm = np.linalg.norm(goal)
            if norm > 0:
                goal = goal / norm

        # Transform goal vec to heading if needed
        if self.goal_observation_format == "heading":
            # heading = np.arctan2(goal[1], goal[0])
            heading = np.arctan2(goal[1], goal[0]) / np.pi  # normalize to [-1, 1]
            goal = np.array([heading], dtype=np.float32)

            # head_deg = np.degrees(heading)
            # print("Goal heading: ", goal, f"({head_deg:.1f} deg)")
        if self.lidar_enabled:
            return {
                "lidar": lidar,
                "goal": goal,
            }
        else:
            return {
                "goal": goal,
            }

    def _compute_reward(self, msgs):
        # Give reward for getting closer to goal
        goal_vector = self._extract_relative_goal_vector(msgs)
        goal_distance = np.linalg.norm(goal_vector)
        reward = 0.0
        if goal_distance < self.best_goal_distance:
            reward += 1 * (self.best_goal_distance - goal_distance)
            # print(f"Reward for getting closer to goal: {reward:.3f}")
            self.best_goal_distance = goal_distance

        if self._get_collision_vel_if_collided(msgs) is not None:
            reward = self.collision_reward
            # print(f"Collision detected! Applying collision reward: {self.collision_reward}")

        return reward

    def _check_terminated(self, msgs):

        if self._get_collision_vel_if_collided(msgs) is not None:
            print("Terminating episode due to collision.")
            return True

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

        # NORMALIZE lidar to max range
        dists = dists / lidar_msg.maxRange
        return dists

    def _extract_goal(self, msgs):
        return np.zeros(2, dtype=np.float32)
