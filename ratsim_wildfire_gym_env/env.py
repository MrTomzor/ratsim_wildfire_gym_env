import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ratsim.roslike_unity_connector.connector import BoolMessage, RoslikeUnityConnector
from ratsim.roslike_unity_connector.message_definitions import (
    Int32Message,
    Float32Message,
    BoolMessage,
    StringMessage,
    TwistMessage,
    PoseMessage,
    Lidar2DMessage,
)
from ratsim.config_blender import to_entries_json
from ratsim.config_blender.blender import flatten_config
from ratsim.transforms import yaw_from_quat
from ratsim.task_tracker import TaskTracker

from ratsim_wildfire_gym_env.curricula import *
from ratsim_wildfire_gym_env.grid_cell_encoder import GridCellEncoder

class WildfireGymEnv(gym.Env):# # #{
    metadata = {"render_modes": []}

    def __init__(
        self,
        worldgen_config: dict,
        agent_config: dict,
        sensor_config: dict,
        action_config: dict,
        task_config: dict,
        metaworldgen_config: dict,
        curriculum_name: str = "",
        episode_log_path: "str | Path | None" = None,
        run_metadata: "dict | None" = None,
        unity_port: int = 9000,
    ):
        super().__init__()
        # unity_port: TCP port the connector should attach to. Default 9000 for
        # backwards compat with single-instance setups. Multi-env training passes
        # a unique port per env (see ratsim.unity_launcher.allocate_unity_instances).
        self.unity_port = unity_port

        self.worldgen_config = worldgen_config
        self.agent_config = agent_config
        self.sensor_config = sensor_config
        self.action_config = action_config
        self.step_count = 0

        # --- Per-episode JSONL logging ---
        # Completed episodes are appended to `episode_log_path` on terminate/truncate
        # with the schema used by test.py's make_episode_result(). `run_metadata`
        # stamps method/rundef/seed/stage_idx into every line.
        # `episode_idx` is made cumulative across stages by counting existing lines
        # in the log file at construction time (each stage constructs a fresh env).
        self.episode_log_path = Path(episode_log_path) if episode_log_path else None
        self.run_metadata = dict(run_metadata) if run_metadata else {}
        self.episode_log_counter = 0
        self.episode_idx_offset = 0
        # Extra key/values merged into each episode's JSONL record. Wrappers may
        # write here (e.g. AdaptiveDifficultyWrapper records 'difficulty').
        self.extra_log_fields: dict = {}
        if self.episode_log_path is not None and self.episode_log_path.exists():
            with open(self.episode_log_path) as f:
                self.episode_idx_offset = sum(1 for _ in f)
        self.episode_start_time = time.time()

        self.curriculum = None
        if curriculum_name != "":
            print(f"Using curriculum: {curriculum_name}")
            self.curriculum = build_curriculum(curriculum_name)
            default_envconfig = self.curriculum.get_default_envconfig()
            default_sensor_config = default_envconfig[1]
            default_action_config = default_envconfig[2]
            default_task_config = default_envconfig[3]

            # If sensor, action or task configs are empty/null, fill them in from curriculum
            if self.sensor_config is None or len(self.sensor_config) == 0:
                self.sensor_config = default_sensor_config
                print("Using curriculum's default sensor config: " + str(self.sensor_config))
            if self.action_config is None or len(self.action_config) == 0:
                self.action_config = default_action_config
                print("Using curriculum's default action config: " + str(self.action_config))
            if task_config is None or len(task_config) == 0:
                task_config = default_task_config
                print("Using curriculum's default task config: " + str(task_config))

        # --- Set sensing params ---
        # TODO - implement if needed
        self.lidar_msg_in_topic = "/lidar2d"
        self.goal_pose_msg_in_topic = "/wildfire_goal_position"
        self.agent_pose_msg_in_topic = "/rat1_pose"
        # Ground-truth pose topic — always-on AbsolutePose2DSensor published by
        # AgentLoader under the agent's name_prefix. Used for reward computation
        # / debugging only (e.g. volumetric exploration); not part of RL obs.
        _flat_agent = flatten_config(self.agent_config)
        _agent_name_prefix = _flat_agent.get("name_prefix", "rat1")
        self.agent_gt_pose_msg_in_topic = f"/{_agent_name_prefix}/gt_pose"

        # --- Task tracker ---
        if task_config is None:
            print("ERROR! no config provided for task tracker, which is required. Please provide a task_config dict with necessary parameters for your task.")
            assert task_config is not None
        # Forward world bounds so the tracker can align a volumetric
        # exploration grid with the maze center.
        _flat_worldgen = flatten_config(self.worldgen_config)
        world_w = _flat_worldgen.get("world_bounds/width", None)
        world_h = _flat_worldgen.get("world_bounds/height", None)
        self.task_tracker = TaskTracker(
            task_config,
            world_width=float(world_w) if world_w is not None else None,
            world_height=float(world_h) if world_h is not None else None,
            pose_topic=self.agent_gt_pose_msg_in_topic,
            lidar_topic=self.lidar_msg_in_topic,
        )
        print("Initialized task tracker with config: " + str(task_config))

        # TODO - metaparams 
        self.num_episodes = 0
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
        # self.gps_normalization_factor = 1
        self.compass_enabled = True
        self.discrete_actions = True

        # --- Grid-cell encoding of GPS ---
        # When True, the "gps" observation becomes a vector of grid-cell
        # activations instead of the raw 2D position. Toggle is set here at
        # env construction; later it can be plumbed through agent_config.
        self.use_grid_cells = False
        self.grid_cell_num_cells = 8
        self.grid_cell_min_scale = 2.0
        self.grid_cell_max_scale = 100.0
        self.grid_cell_seed = 0
        self.grid_cell_verbose = False
        self.grid_cell_encoder = None
        if self.use_grid_cells:
            self.grid_cell_encoder = GridCellEncoder(
                num_cells=self.grid_cell_num_cells,
                min_scale=self.grid_cell_min_scale,
                max_scale=self.grid_cell_max_scale,
                seed=self.grid_cell_seed,
            )
            print(
                f"Grid-cell encoding enabled: num_cells={self.grid_cell_num_cells}, "
                f"min_scale={self.grid_cell_min_scale}, max_scale={self.grid_cell_max_scale}"
            )

        # --- Sector signal sensor (driven by agent_config) ---
        # agent_config may be in structured form (sensors: [ {name: ..., ...}, ... ]);
        # flatten to the same flat dict Unity receives so we can query by flat keys.
        _flat_cfg = flatten_config(self.agent_config)
        _sensors_list = [s.strip() for s in str(_flat_cfg.get("sensors", "")).split(",")]
        print("SENSORS LIST:", _sensors_list)
        self.sector_signal_enabled = "sector_signal" in _sensors_list
        if self.sector_signal_enabled:
            _ch_str = str(_flat_cfg.get("sector_signal/channels", "default"))
            self.sector_signal_channels = [c.strip() for c in _ch_str.split(",") if c.strip()]
            self.sector_signal_n_sectors = int(_flat_cfg.get("sector_signal/nSectors", 8))
            _prefix = str(_flat_cfg.get("sector_signal/topicPrefix", "/sector_signal"))
            self.sector_signal_topics = [f"{_prefix}/{c}" for c in self.sector_signal_channels]
            print(f"Sector signal enabled: channels={self.sector_signal_channels}, n_sectors={self.sector_signal_n_sectors}")

        # --- Connect to Ratsim ---
        self.conn = RoslikeUnityConnector(port=self.unity_port, verbose=False)
        self.conn.connect()

        # --- Select scene ---
        self._select_scene("Wildfire")
        print("Selected Wildfire scene in Ratsim.")

        # --- Send agent config ---
        agent_config_json = to_entries_json(self.agent_config)
        print("Sending agent config: " + agent_config_json)
        self.conn.publish(StringMessage(data=agent_config_json), "/sim_control/agent_config")
        self.conn.send_messages_and_step(enable_physics_step=False)
        self.conn.read_messages_from_unity()

        # Random generaiton preparation
        # Create objects necessary for generating a world seed using the metaseed on every reset
        self.metaworldgen_config = metaworldgen_config
        self.world_seed_generator = None
        if self.metaworldgen_config is not None and "world_generation_metaseed" in self.metaworldgen_config:
            seed_generation_seed = metaworldgen_config["world_generation_metaseed"]
            self.world_seed_generator = np.random.default_rng(seed_generation_seed)

        self.reset()
        print("Did first reset to prepare environment and observation topics.")

        # -- Get observation topics ready by doing one step of communication
        self.conn.send_messages_and_step(enable_physics_step=False)
        msgs = self.conn.read_messages_from_unity()

        lidar_msg = msgs[self.lidar_msg_in_topic][0]
        if lidar_msg is None:
            print("ERROR! no lidar message received during initialization. Check connection and topic names.")
        num_lidar_rays = len(lidar_msg.ranges)
        num_lidar_semantics = len(lidar_msg.descriptors) if lidar_msg.descriptors is not None else 0
        print("Number of lidar rays: " + str(num_lidar_rays) + ", number of semantic channels: " + str(num_lidar_semantics))
        self.num_lidar_rays = num_lidar_rays
        self.num_lidar_channels = (num_lidar_semantics // num_lidar_rays + 1) if self.lidar_observation_format == "depth_and_semantics" else 1



        # --- Set up action params, read from configs ---
        # self.max_forward_velocity = self.action_config.get("max_forward_velocity", 20.0)
        # self.max_angular_velocity = self.action_config.get("max_angular_velocity", 1.0)
        self.max_forward_velocity = self.action_config.get("max_forward_velocity", 10.0)
        self.max_angular_velocity = self.action_config.get("max_angular_velocity", 1.5)
        self.max_forward_acceleration = self.action_config.get("max_forward_acceleration", 100)
        self.max_angular_acceleration = self.action_config.get("max_angular_acceleration", 13)
        self.vel_twist_msg_out_topic = "/cmd_vel"
        self.accel_twist_msg_out_topic = "/cmd_accel"
        # self.control_mode = self.action_config.get("control_mode", "velocity") 
        self.control_mode = self.action_config.get("control_mode") 
        print(f"Action config: max_forward_velocity={self.max_forward_velocity}, max_angular_velocity={self.max_angular_velocity}")

        # --- Action space ---
        if self.discrete_actions:
            # 3 linear (backward, stop, forward) x 5 angular (hard left .. hard right)
            self._linear_bins = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
            self._angular_bins = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
            self.action_space = spaces.MultiDiscrete([len(self._linear_bins), len(self._angular_bins)])
        else:
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

        # Get gps enabled and compass enabled from agent config
        # for gps - agent should have either relative_pose or odom in the list of sensors
        self.gps_enabled = any(s in _sensors_list for s in ["relative_pose", "odom"])
        self.compass_enabled = "compass" in _sensors_list
        print("GPS enabled: " + str(self.gps_enabled) + ", Compass enabled: " + str(self.compass_enabled))

        if self.compass_enabled:
            obsvdict["compass"] = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

        if self.gps_enabled:
            if self.use_grid_cells:
                obsvdict["gps"] = spaces.Box(
                    0.0, 1.0, shape=(self.grid_cell_num_cells,), dtype=np.float32
                )
            else:
                # TODO - handle size correctly based on worldgen config (arena size)?
                obsvdict["gps"] = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

        if self.sector_signal_enabled:
            _sig_total = len(self.sector_signal_channels) * self.sector_signal_n_sectors
            obsvdict["sector_signal"] = spaces.Box(0.0, 1.0, shape=(_sig_total,), dtype=np.float32)

        self.observation_space = spaces.Dict(obsvdict)

        # print dimensionality of observation space
        print("Observation space:", self.observation_space)


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
        # TODO - reimplement curriculum functionality with blender
        # if self.curriculum is not None:
        #     worldgen_config, is_new_chapter = self.curriculum.get_worldconfig_for_episode(self.num_episodes)
        #     if is_new_chapter:
        #         print(f"Curriculum advanced to new chapter at episode {self.num_episodes}!")
        #         print(f"New worldgen config: {worldgen_config}")
        #     self.worldgen_config = worldgen_config

        # Change seed if using metaworldgen
        if self.world_seed_generator is not None:
            if options is None:
                options = {}
            new_world_seed = int(self.world_seed_generator.integers(0, 2**31 - 1))
            print(f"Generated new world seed: {new_world_seed}")
            # worldgen_msgs.seed = new_world_seed
            options["seed"] = new_world_seed

        # --- World generation ---
        # Send config as unified JSON via new config pipeline
        cfg = dict(self.worldgen_config)
        if options is not None:
            cfg.update(options)
        config_json = to_entries_json(cfg)
        # print("Sending json worldgen config: " + config_json)
        self.conn.publish(StringMessage(data=config_json), "/sim_control/world_config")
        self.conn.publish(BoolMessage(data=True), "/sim_control/reset_episode")
        self.conn.send_messages_and_step(enable_physics_step=True)
        obs_msgs = self.conn.read_messages_from_unity()
        # Surface worldgen errors/warnings posted during the reset step.
        self.conn.process_worldgen_status()

        # Perform one more step to let worldgen happen
        self.conn.send_messages_and_step(enable_physics_step=True)

        # --- Read initial observations ---
        obs_msgs = self.conn.read_messages_from_unity()
        self.conn.process_worldgen_status()
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

        # Store completed episode metrics before clearing
        if hasattr(self, 'distance_traveled'):
            self._finalize_episode_metrics()

        self.reset_logging_metrics()
        self.task_tracker.reset()
        self.episode_start_time = time.time()
        print("Reset complete, returning initial observation.")


        return obs, {}
# # #}

    def step(self, action):# # #{
        self.step_count += 1
        # print("Step count: " + str(self.step_count))

        # --- Send action ---
        action_msg = self._build_action_msg(action)
        # print("Action message to send: linear_x " + str(action_msg.linear_x) + ", angular_z " + str(action_msg.angular_z))
        # print("Control mode: " + str(self.control_mode))
        if self.control_mode == "acceleration":
            self.conn.publish(action_msg, self.accel_twist_msg_out_topic)
        else:
            self.conn.publish(action_msg, self.vel_twist_msg_out_topic)

        self.conn.send_messages_and_step(enable_physics_step=True)
        msgs = self.conn.read_messages_from_unity()

        # self.conn.log_connection_stats()

        obs = self._parse_observation(msgs)
        self.task_tracker.update_with_unity_msgs(msgs)
        reward = self.task_tracker.get_this_step_score()
        terminated = self.task_tracker.is_terminated()
        truncated = self.step_count >= self.task_tracker.episode_max_steps
        if truncated:
            print("Truncating episode due to max steps reached: " + str(self.task_tracker.episode_max_steps))
        if terminated:
            print("Terminating episode at step " + str(self.step_count) + " with reward " + str(reward)
                  + " (reason: " + str(self.task_tracker.get_termination_reason()) + ")")

        # Log metrics for debugging
        self._parse_and_log_metrics(msgs)

        info = {}
        if terminated or truncated:
            self.task_tracker.print_exploration_summary(prefix="end-of-episode")
            self._log_episode_jsonl(terminated=terminated, truncated=truncated)
            # Surface end-of-episode stats for wrappers (e.g. adaptive difficulty
            # reads episode_pickups to decide success/failure).
            info["episode_pickups"] = self.task_tracker.get_num_reward_objs_picked_up()
            info["termination_reason"] = self.task_tracker.get_termination_reason()

        return obs, reward, terminated, truncated, info
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
        msg = TwistMessage()

        if self.discrete_actions:
            linear_norm = float(self._linear_bins[action[0]])
            angular_norm = float(self._angular_bins[action[1]])
        else:
            linear_norm = float(action[0])
            angular_norm = float(action[1])

        if self.control_mode == "acceleration":
            msg.linear_x = linear_norm * self.max_forward_acceleration
            msg.linear_y = 0
            msg.linear_z = 0
            msg.angular_x = 0
            msg.angular_y = 0
            msg.angular_z = angular_norm * self.max_angular_acceleration
        else:
            msg.linear_x = linear_norm * self.max_forward_velocity
            self.largest_forward_velocity = max(self.largest_forward_velocity, abs(msg.linear_x))
            msg.linear_y = 0
            msg.linear_z = 0
            msg.angular_x = 0
            msg.angular_y = 0
            msg.angular_z = angular_norm * self.max_angular_velocity
        return msg
# # #}


    def _extract_relative_goal_vector(self, msgs):# # #{
        res = np.zeros(2, dtype=np.float32)

        if not self.goal_pose_msg_in_topic in msgs.keys() or not self.agent_pose_msg_in_topic in msgs.keys():
            # print("Warning: goal or agent pose message not found in msgs.")
            # print("Which topics are available:", list(msgs.keys()))
            return res

        goal_msg = msgs[self.goal_pose_msg_in_topic][0]
        agent_msg = msgs[self.agent_pose_msg_in_topic][0]

        gx = goal_msg.x
        gy = goal_msg.y

        ax = agent_msg.x
        ay = agent_msg.y
        a_theta = yaw_from_quat(agent_msg.qx, agent_msg.qy, agent_msg.qz, agent_msg.qw)

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


    # --- Logging helpers ---

    def reset_logging_metrics(self):# # #{
        self.last_logged_position = None
        self.distance_traveled = 0.0
        self.longest_step_distance = 0.0
        self.largest_forward_velocity = 0.0
        if not hasattr(self, '_completed_episode_distances'):
            self._completed_episode_distances = []
            self._completed_episode_pickups = []
            self._completed_episode_explored_area = []
        # # #}

    def _finalize_episode_metrics(self):# # #{
        """Store completed episode metrics before reset clears them."""
        self._completed_episode_distances.append(self.distance_traveled)
        self._completed_episode_pickups.append(self.task_tracker.get_num_reward_objs_picked_up())
        self._completed_episode_explored_area.append(float(self.task_tracker.get_explored_area_m2()))
        # # #}

    def _parse_and_log_metrics(self, msgs):# # #{
        odom_topic = "/odom"
        if odom_topic in msgs.keys():
            odom_msg = msgs[odom_topic][0]
            deltavec_in_prev_frame = np.array([odom_msg.x, odom_msg.y])

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
        return self.task_tracker.get_num_reward_objs_picked_up()
    # # #}

    def get_completed_episode_distances(self):# # #{
        """Return and clear the list of completed episode distances."""
        result = list(self._completed_episode_distances)
        self._completed_episode_distances.clear()
        return result
    # # #}

    def get_completed_episode_pickups(self):# # #{
        """Return and clear the list of completed episode pickups."""
        result = list(self._completed_episode_pickups)
        self._completed_episode_pickups.clear()
        return result
    # # #}

    def get_completed_episode_explored_area(self):# # #{
        """Return and clear the list of completed episode explored areas (m²)."""
        result = list(self._completed_episode_explored_area)
        self._completed_episode_explored_area.clear()
        return result
    # # #}

    def get_longest_step_distance(self):# # #{
        return self.longest_step_distance
    # # #}

    def get_num_episodes(self):# # #{
        """Total completed episodes written to JSONL in this env instance."""
        return self.episode_log_counter
    # # #}

    def _log_episode_jsonl(self, terminated: bool, truncated: bool):# # #{
        """Append one JSON line per completed episode, matching test.py's schema."""
        if self.episode_log_path is None:
            return
        self.episode_log_counter += 1
        reason = self.task_tracker.get_termination_reason()
        if reason is None:
            reason = "max_steps" if truncated else "unknown"
        record = {
            "method": self.run_metadata.get("method"),
            "rundef": self.run_metadata.get("rundef"),
            "stage_idx": self.run_metadata.get("stage_idx"),
            "seed": self.run_metadata.get("seed"),
            "episode_idx": self.episode_idx_offset + self.episode_log_counter,
            "steps": self.step_count,
            "total_score": self.task_tracker.get_total_score(),
            "objects_found": self.task_tracker.get_num_reward_objs_picked_up(),
            "collisions": self.task_tracker.get_collision_count(),
            "termination_reason": reason,
            "distance_traveled": float(self.distance_traveled),
            "explored_area_m2": float(self.task_tracker.get_explored_area_m2()),
            "wall_time_s": time.time() - self.episode_start_time,
            **self.extra_log_fields,
        }
        self.episode_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.episode_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
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

        if self.sector_signal_enabled:
            resdict["sector_signal"] = self._extract_sector_signal(msgs)

        return resdict
# # #}

    def _extract_lidar(self, msgs):# # #{
        # TODO: parse lidar topic, just the distances
        if not self.lidar_msg_in_topic in msgs.keys():
            print("Warning: lidar message not found in msgs.")
            print("Which topics are available:", list(msgs.keys()))
            # return np.zeros(self.observation_space["lidar"].shape, dtype=np.float32)
            return None
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
        if self.use_grid_cells:
            zero_out = np.zeros(self.grid_cell_num_cells, dtype=np.float32)
        else:
            zero_out = np.zeros(2, dtype=np.float32)
        if not pose_relative_to_start_topic in msgs.keys():
            print("Warning: GPS message not found in msgs.")
            print("Which topics are available:", list(msgs.keys()))
            return zero_out
        pose_msg = msgs[pose_relative_to_start_topic][0]

        if self.use_grid_cells:
            # Encode raw (un-normalized) position so grid-cell scales are
            # interpreted in arena meters.
            activations = self.grid_cell_encoder.encode(pose_msg.x, pose_msg.y)
            if self.grid_cell_verbose:
                with np.printoptions(precision=3, suppress=True, linewidth=200):
                    print(
                        f"Grid-cell activations @ ({pose_msg.x:.2f}, {pose_msg.y:.2f}): "
                        f"{activations}"
                    )
            return activations

        res = np.zeros(2, dtype=np.float32)
        res[0] = pose_msg.x / self.gps_normalization_factor
        res[1] = pose_msg.y / self.gps_normalization_factor
        if np.abs(res[0]) > 1.0 or np.abs(res[1]) > 1.0:
            print("Warning: GPS reading exceeds normalization bounds, consider increasing normalization factor.")
        return res# # #}

    def _extract_compass(self, msgs):# # #{
        pose_relative_to_start_topic = "/rat1_pose_from_start"
        res = np.zeros(1, dtype=np.float32)
        if not pose_relative_to_start_topic in msgs.keys():
            print("Warning: Compass message not found in msgs.")
            print("Which topics are available:", list(msgs.keys()))
            return res
        pose_msg = msgs[pose_relative_to_start_topic][0]
        yaw = yaw_from_quat(pose_msg.qx, pose_msg.qy, pose_msg.qz, pose_msg.qw)
        heading = yaw / np.pi  # normalize to [-1, 1]

        res[0] = heading
        # print("Compass reading (normalized heading): " + str(res))
        return res
# # #}

    def _extract_goal(self, msgs):# # #{
        return np.zeros(2, dtype=np.float32)# # #}

    def _extract_sector_signal(self, msgs):# # #{
        # Flattens [channel0_sec0, channel0_sec1, ..., channel1_sec0, ...] in the same
        # channel order as self.sector_signal_channels. Missing topics -> zeros.
        n_ch = len(self.sector_signal_channels)
        n_s = self.sector_signal_n_sectors
        out = np.zeros(n_ch * n_s, dtype=np.float32)
        for ci, topic in enumerate(self.sector_signal_topics):
            if topic not in msgs:
                continue
            msg = msgs[topic][0]
            if msg is None or msg.data is None:
                continue
            arr = np.asarray(msg.data, dtype=np.float32)
            k = min(n_s, len(arr))
            out[ci * n_s : ci * n_s + k] = arr[:k]
        return out
    # # #}

