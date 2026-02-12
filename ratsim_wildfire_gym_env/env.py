import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ratsim.roslike_unity_connector.connector import BoolMessage, RoslikeUnityConnector
from ratsim.roslike_unity_connector.message_definitions import (
    # WildfireWorldGenMessage,
    Int32Message,
    Float32Message,
    BoolMessage,
    StringMessage,
    Twist2DMessage,
    Lidar2DMessage,
)

from ratsim_wildfire_gym_env.curricula import *

class WildfireGymEnv(gym.Env):# # #{
    metadata = {"render_modes": []}

    def __init__(
        self,
        worldgen_config: dict,
        sensor_config: dict,
        action_config: dict,
        reward_config: dict,
        metaworldgen_config: dict,
        curriculum_name: str = "",
    ):
        super().__init__()

        self.worldgen_config = worldgen_config
        self.sensor_config = sensor_config
        self.action_config = action_config
        self.reward_config = reward_config
        self.step_count = 0

        self.curriculum = None
        if curriculum_name != "":
            print(f"Using curriculum: {curriculum_name}")
            self.curriculum = build_curriculum(curriculum_name)
            default_envconfig = self.curriculum.get_default_envconfig()
            default_sensor_config = default_envconfig[1]
            default_action_config = default_envconfig[2]
            default_reward_config = default_envconfig[3]

            # If sensor, action or reward configs are empty/null, fill them in from curriculum
            if self.sensor_config is None or len(self.sensor_config) == 0:
                self.sensor_config = default_sensor_config
                print("Using curriculum's default sensor config: " + str(self.sensor_config))
            if self.action_config is None or len(self.action_config) == 0:
                self.action_config = default_action_config
                print("Using curriculum's default action config: " + str(self.action_config))
            if self.reward_config is None or len(self.reward_config) == 0:
                self.reward_config = default_reward_config
                print("Using curriculum's default reward config: " + str(self.reward_config))

        # --- Set sensing params ---
        # TODO - implement if needed
        self.lidar_msg_in_topic = "/lidar2d"
        self.goal_pose_msg_in_topic = "/wildfire_goal_position"
        self.agent_pose_msg_in_topic = "/rat1_pose"

        # TODO - metaparams 
        self.goal_observation_format = "none" # or deltavec, or heading
        # self.goal_observation_format = "normalized_deltavec" # or deltavec, or heading
        # self.goal_observation_format = "deltavec" # or deltavec, or heading
        # self.goal_observation_format = "heading" # or deltavec, or heading
        # self.lidar_observation_format = "depth_only"
        self.lidar_observation_format = "depth_and_semantics"
        self.lidar_enabled = True
        # TODO - handle gps normalization factor based on worldgen config (arena size)?
        self.gps_enabled = True
        self.gps_normalization_factor = 300.0 # divide gps readings by this factor to keep in reasonable range for NN
        self.compass_enabled = True

        # --- Connect to Ratsim ---
        self.conn = RoslikeUnityConnector(verbose=False)
        self.conn.connect()

        # --- Select scene ---
        self._select_scene("Wildfire")
        print("Selected Wildfire scene in Ratsim.")

        # -- Get observation topics ready by doing one step of communication
        self.conn.send_messages_and_step(enable_physics_step=False)
        msgs = self.conn.read_messages_from_unity()

        lidar_msg = msgs[self.lidar_msg_in_topic][0]
        if lidar_msg is None:
            print("ERROR! no lidar message received during initialization. Check connection and topic names.")
        num_lidar_rays = len(lidar_msg.ranges) 
        num_lidar_semantics = len(lidar_msg.descriptors) if lidar_msg.descriptors is not None else 0



        # --- Set up action params, read from configs ---
        # self.max_forward_velocity = self.action_config.get("max_forward_velocity", 20.0)
        # self.max_angular_velocity = self.action_config.get("max_angular_velocity", 1.0)
        self.max_forward_velocity = self.action_config.get("max_forward_velocity", 10.0)
        self.max_angular_velocity = self.action_config.get("max_angular_velocity", 0.5)
        self.max_forward_acceleration = self.action_config.get("max_forward_acceleration", 10.0)
        self.max_angular_velocity = self.action_config.get("max_forward_acceleration", 10.0)
        self.vel_twist_msg_out_topic = "/cmd_vel"
        self.accel_twist_msg_out_topic = "/cmd_accel"
        # self.control_mode = self.action_config.get("control_mode", "velocity") 
        self.control_mode = self.action_config.get("control_mode") 
        print(f"Action config: max_forward_velocity={self.max_forward_velocity}, max_angular_velocity={self.max_angular_velocity}")

        # --- Action space ---
        # Example: differential drive
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

        # --- Observation space ---
        # Example: lidar + goal vector
        obsvdict = {}
        if self.lidar_enabled:
            if self.lidar_observation_format == "depth_only":
                obsvdict["lidar"] = spaces.Box(0.0, 1.0, shape=(num_lidar_rays,), dtype=np.float32) # normalized
            elif self.lidar_observation_format == "depth_and_semantics":
                obsvdict["lidar"] = spaces.Box(0.0, 1.0, shape=(num_lidar_semantics + num_lidar_rays,), dtype=np.float32)
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
        elif self.goal_observation_format == "none":
            pass
        else:
            print("Warning: unknown goal observation format, defaulting to normalized_deltavec")
            return 1

        if self.compass_enabled:
            obsvdict["compass"] = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

        if self.gps_enabled:
            # TODO - handle size correctly based on worldgen config (arena size)?
            obsvdict["gps"] = spaces.Box(-1, 1, shape=(2,), dtype=np.float32) 

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

        self.num_episodes = 0
        self.reset_logging_metrics()
# # #}

    # ---------------------------
    # Gym API
    # ---------------------------

    def reset(self, *, seed=None, options=None):# # #{
        super().reset(seed=seed)
        self.step_count = 0
        self.num_episodes += 1

        # if self.num_episodes == 200:
        #     print("MANY EPISODES, SWITCHING TO BIGGER ENV!")
        #     self.worldgen_config["arenaWidth"] *= 10
        #     self.worldgen_config["arenaHeight"] *= 10

        print("Resetting environment. Episode number: " + str(self.num_episodes))

        # If curriculum is being used, update worldgen config based on curriculum progression
        if self.curriculum is not None:
            worldgen_config, is_new_chapter = self.curriculum.get_worldconfig_for_episode(self.num_episodes)
            if is_new_chapter:
                print(f"Curriculum advanced to new chapter at episode {self.num_episodes}!")
                print(f"New worldgen config: {worldgen_config}")
            self.worldgen_config = worldgen_config

        # Change seed if using metaworldgen
        if self.world_seed_generator is not None:
            if options is None:
                options = {}
            new_world_seed = int(self.world_seed_generator.integers(0, 2**31 - 1))
            print(f"Generated new world seed: {new_world_seed}")
            # worldgen_msgs.seed = new_world_seed
            options["seed"] = new_world_seed

        # --- World generation ---
        worldgen_msgs = self._build_worldgen_msgs(options)
        worldgen_msgs["requested"] = BoolMessage(data=True)

        # self.conn.publish(worldgen_msg, "/wildfire_worldgen_input")
        for topic, msg in worldgen_msgs.items():
            # print(f"Publishing worldgen msg on topic /worldgen/{topic}: {msg}")
            self.conn.publish(msg, f"/worldgen/{topic}")
        self.conn.send_messages_and_step(enable_physics_step=True)
        obs_msgs = self.conn.read_messages_from_unity()

        # Perform one more step to let worldgen happen
        self.conn.send_messages_and_step(enable_physics_step=True)

        # --- Read initial observations ---
        obs_msgs = self.conn.read_messages_from_unity()
        obs = self._parse_observation(obs_msgs)

        # -- Reset goal nearing tracking ---

        # Write initial best goal distance if was initialized before
        if hasattr(self, 'best_goal_distance'):
            print("best goal distance:" + str(self.best_goal_distance))
        goal_vector = self._extract_relative_goal_vector(obs_msgs)
        self.best_goal_distance = np.linalg.norm(goal_vector)

        if hasattr(self, 'longest_step_distance'):
            print("largest step and forward velocity:")
            print(self.longest_step_distance)
            print(self.largest_forward_velocity)

        self.reset_logging_metrics()


        return obs, {}
# # #}

    def step(self, action):# # #{
        self.step_count += 1
        # print("Step count: " + str(self.step_count))

        # --- Send action ---
        action_msg = self._build_action_msg(action)
        if self.control_mode == "acceleration":
            self.conn.publish(action_msg, self.accel_twist_msg_out_topic)
        else:
            self.conn.publish(action_msg, self.vel_twist_msg_out_topic)

        self.conn.send_messages_and_step(enable_physics_step=True)
        msgs = self.conn.read_messages_from_unity()

        # self.conn.log_connection_stats()

        obs = self._parse_observation(msgs)
        reward = self._compute_reward(msgs)
        terminated = self._check_terminated(msgs)
        max_steps = self.reward_config.get("max_steps", 1000)
        truncated = self.step_count >= max_steps
        if truncated:
            print("Truncating episode due to max steps reached: " + str(max_steps))
        if terminated:
            print("Terminating episode at step " + str(self.step_count) + " with reward " + str(reward))

        # Log metrics for debugging
        self._parse_and_log_metrics(msgs)

        return obs, reward, terminated, truncated, {}
# # #}

    def close(self):# # #{
        # self.conn.close()
        print("Closing WildfireGymEnv.")
        return
# # #}

    # ---------------------------
    # Helpers
    # ---------------------------

    def _select_scene(self, scene_name: str):# # #{
        msg = StringMessage(data=scene_name)
        self.conn.publish(msg, "/sim_control/scene_select")
        self.conn.send_messages_and_step(enable_physics_step=False)
        self.conn.read_messages_from_unity()
# # #}

    def _build_worldgen_msgs(self, options):# # #{
        cfg = dict(self.worldgen_config)
        if options is not None:
            cfg.update(options)

        # transform the dict of name : value into a dict of topic(=name) : message for basic data types
        worldgen_msgs = {}
        for k, v in cfg.items():
            if isinstance(v, bool):
                msg = BoolMessage(data=v)
            elif isinstance(v, int):
                msg = Int32Message(data=v)
            elif isinstance(v, float):
                msg = Float32Message(data=v)
            elif isinstance(v, str):
                msg = StringMessage(data=v)
            else:
                print(f"Warning: unsupported worldgen config type for key {k}, value {v}. Skipping.")
                continue
            worldgen_msgs[k] = msg
        return worldgen_msgs

        # msg = WildfireWorldGenMessage()
        # for k, v in cfg.items():
        #     setattr(msg, k, v)
        # return msg
# # #}

    def _build_action_msg(self, action):# # #{
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
            msg.forward = 10 * np.sign(msg.forward) #debug 
            self.largest_forward_velocity = max(self.largest_forward_velocity , abs(msg.forward))
            # print("forward velocity: " + str(msg.forward))
            msg.left = 0
            msg.radiansCounterClockwise = float(action[1]) * self.max_angular_velocity
            # msg.radiansCounterClockwise = 0 #debug
        return msg
# # #}

    def _check_num_reward_objs_picked_up(self, msgs):# # #{
        rew_topic = "/reward_pickup"
        if not rew_topic in msgs.keys():
            return 0
        # its int msgs, each corresponds to some objects numerosity
        rew_msgs = msgs[rew_topic]
        total_picked_up = sum([msg.data for msg in rew_msgs])
        return total_picked_up

# # #}

    def _extract_relative_goal_vector(self, msgs):# # #{
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
# # #}

    def _compute_reward(self, msgs):# # #{
        # Give reward for getting closer to goal
        goal_vector = self._extract_relative_goal_vector(msgs)
        reward = 0.0
        # goal_distance = np.linalg.norm(goal_vector)
        # if goal_distance < self.best_goal_distance:
        #     reward += 1 * (self.best_goal_distance - goal_distance)
        #     # print(f"Reward for getting closer to goal: {reward:.3f}")
        #     self.best_goal_distance = goal_distance

        if self._get_collision_vel_if_collided(msgs) is not None:
            reward += self.reward_config.get("hard_collision_reward", -100.0)
            self.collision_count += 1
            # print(f"Collision detected! Applying collision reward: {self.collision_reward}")
            print("COLLISION VEL: " + str(self._get_collision_vel_if_collided(msgs)))

        num_pickup_objects = self._check_num_reward_objs_picked_up(msgs)
        pickupable_reward = num_pickup_objects * self.reward_config.get("reward_pickup_reward", 20.0)
        self.num_reward_objs_picked_up += num_pickup_objects
        reward += pickupable_reward

        if(pickupable_reward > 0):
            print(f"!!! - Reward for picking up objects: {pickupable_reward}")

        return reward
# # #}

    def _check_terminated(self, msgs):# # #{

        if self._get_collision_vel_if_collided(msgs) is not None:
            self.collided_ended = True
            print("Terminating episode due to collision.")
            return True

        return False
# # #}

    # --- Logging helpers ---

    def reset_logging_metrics(self):# # #{
        self.last_logged_position = None
        self.distance_traveled = 0.0
        self.longest_step_distance = 0.0
        self.largest_forward_velocity = 0.0
        self.num_reward_objs_picked_up = 0.0
        self.collision_count = 0
        # # #}

    def _parse_and_log_metrics(self, msgs):# # #{
        odom_topic = "/odom"
        if odom_topic in msgs.keys():
            odom_msg = Twist2DMessage()
            odom_msg = msgs[odom_topic][0]
            deltavec_in_prev_frame = np.array([odom_msg.forward, odom_msg.left])

            # step_distance = np.linalg.norm(current_pos - self.last_logged_position)
            step_distance = np.linalg.norm(deltavec_in_prev_frame)
            # print("Step distance traveled: " + str(step_distance))
            if step_distance > self.longest_step_distance:
                self.longest_step_distance = step_distance
            self.distance_traveled += step_distance
            # print("ADDED DISTANCE: " + str(step_distance) + ", TOTAL: " + str(self.distance_traveled))
    # # #}

    def get_distance_traveled(self):# # #{
        return self.distance_traveled
    # # #}

    def get_reward_pickups(self):# # #{
        return self.num_reward_objs_picked_up
    # # #}

    def get_longest_step_distance(self):# # #{
        return self.longest_step_distance
    # # #}

    # --- Sensor helpers ---
    def _parse_observation(self, msgs):# # #{
        # This is where your sensor parsing lives
        lidar = self._extract_lidar(msgs)
        # print("Lidar sample:")
        # print(lidar)
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
        resdict = {}
        if self.lidar_enabled:
            resdict["lidar"] = lidar
        if not self.goal_observation_format == "none":
            resdict["goal"] = goal

        if self.compass_enabled:
            compass = self._extract_compass(msgs)
            resdict["compass"] = compass

        if self.gps_enabled:
            gps = self._extract_gps(msgs)
            resdict["gps"] = gps


        return resdict
# # #}

    def _get_collision_vel_if_collided(self, msgs):# # #{
        collision_topic = "/collisions"
        if not collision_topic in msgs.keys():
            return None
        collision_msg = msgs[collision_topic][0]
        return collision_msg.data
# # #}

    def _extract_lidar(self, msgs):# # #{
        # TODO: parse lidar topic, just the distances
        if not self.lidar_msg_in_topic in msgs.keys():
            print("Warning: lidar message not found in msgs.")
            print("Which topics are available:", list(msgs.keys()))
            return np.zeros(self.observation_space["lidar"].shape, dtype=np.float32)
        lidar_msg = msgs[self.lidar_msg_in_topic][0]

        # Lidar has -1 for out of range, convert to max range
        dists = np.array(lidar_msg.ranges, dtype=np.float32)
        dists[dists < 0] = lidar_msg.maxRange

        # NORMALIZE lidar to 0-1
        dists = dists / lidar_msg.maxRange

        mean_dist = np.mean(dists)
        # print("Lidar mean distance: " + str(mean_dist))

        if lidar_msg.descriptors is not None and self.lidar_observation_format == "depth_and_semantics":
            semantics = np.array(lidar_msg.descriptors, dtype=np.float32)
            # one hot encoded so no normalization needed
            return np.concatenate([dists, semantics], axis=0)

        return dists# # #}

    def _extract_gps(self, msgs):# # #{
        pose_relative_to_start_topic = "/rat1_pose_from_start"
        res = np.zeros(2, dtype=np.float32)
        if not pose_relative_to_start_topic in msgs.keys():
            print("Warning: GPS message not found in msgs.")
            print("Which topics are available:", list(msgs.keys()))
            return res
        pose_msg = msgs[pose_relative_to_start_topic][0]
        res[0] = pose_msg.forward / self.gps_normalization_factor
        res[1] = pose_msg.left / self.gps_normalization_factor
        if np.abs(res[0]) > 1.0 or np.abs(res[1]) > 1.0:
            print("Warning: GPS reading exceeds normalization bounds, consider increasing normalization factor.")
        # print("GPS reading (normalized): " + str(res))
        return res# # #}

    def _extract_compass(self, msgs):# # #{
        pose_relative_to_start_topic = "/rat1_pose_from_start"
        res = np.zeros(1, dtype=np.float32)
        if not pose_relative_to_start_topic in msgs.keys():
            print("Warning: Compass message not found in msgs.")
            print("Which topics are available:", list(msgs.keys()))
            return res
        pose_msg = msgs[pose_relative_to_start_topic][0]
        # heading = pose_msg.radiansCounterClockwise / np.pi # normalize to [-1, 1]
        # the input data can be outsie of [-pi, pi] range due to how we handle rotation in ratsim, so we need to wrap it to that range before normalizing
        heading = np.arctan2(np.sin(pose_msg.radiansCounterClockwise), np.cos(pose_msg.radiansCounterClockwise)) / np.pi

        res[0] = heading
        # print("Compass reading (normalized heading): " + str(res))
        return res
# # #}

    def _extract_goal(self, msgs):# # #{
        return np.zeros(2, dtype=np.float32)# # #}

