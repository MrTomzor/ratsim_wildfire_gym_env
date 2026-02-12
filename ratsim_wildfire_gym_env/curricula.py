import numpy as np

class Curriculum:# # #{
    def __init__(self, cw = [], cl = []):
        self.chapters_envconfigs = cw
        self.chapter_lengths = cl

    def get_worldconfig_for_episode(self, episode_num):
        is_new_chapter = False

        cumulative_length = 0
        for i, chapter_length in enumerate(self.chapter_lengths):
            cumulative_length += chapter_length
            if episode_num < cumulative_length:
                if episode_num == cumulative_length - chapter_length:
                    is_new_chapter = True
                return self.chapters_envconfigs[i][0], is_new_chapter

        print("curriculum episode number exceeds total length, returning last chapter's worldconfig")
        return self.chapters_envconfigs[-1], is_new_chapter

    def get_total_length(self):
        return sum(self.chapter_lengths)

    def get_default_envconfig(self):
        # just return the configs from the first chapter, since they are not expected to change across chapters
        if len(self.chapters_envconfigs) == 0:
            raise ValueError("Curriculum has no chapters, cannot get default configs")
        return self.chapters_envconfigs[0]
# # #}

def get_named_envconfig(name):# # #{
    worldgen_config = {}
    sensor_config = {}

    reward_config = {
        "hard_collision_reward" : -100,
        "reward_pickup_reward" : 20,
    }

    action_config = {
        # "control_mode": "acceleration",
        "control_mode": "velocity",
    }

    if name == "forest_foraging_easy_1":
        worldgen_config = {
            "seed": 42, # will be overridden by metaworldgen_config
            "mainLayout" : "suburb",
            "numAgents": 1,
            "startAndGoalClearingDistance": 5.0,
            "arenaWidth": 300.0, # have to be float for proper msg conversion
            "arenaHeight": 300.0,
            "treeDensity": 0.01,
            "treeOscillationEnabled" : False,
            "houseNumerosity" : 0.0,
            "houseDoorDefaultType" : "none",
            "rewardNumerosity" : 0.005,
            "rewardDistribution" : "everywhere",
        }
        reward_config["max_steps"] = 1000

    elif name == "houses_only_easy_1":
        worldgen_config = {
            "seed": 42, # will be overridden by metaworldgen_config
            "mainLayout" : "suburb",
            "numAgents": 1,
            "startAndGoalClearingDistance": 5.0,
            "arenaWidth": 150.0, # have to be float for proper msg conversion
            "arenaHeight": 150.0,
            "treeDensity": 0.00,
            "treeOscillationEnabled" : False,
            "houseNumerosity" : 20.0,
            "houseDoorDefaultType" : "none",
            "rewardNumerosity" : 1.0,
            "rewardDistribution" : "houses",
        }
        reward_config["max_steps"] = 2000

        pass
    else:
        raise ValueError(f"Unknown curriculum name: {name}")

    return worldgen_config, sensor_config, action_config, reward_config
# # #}

def build_curriculum(name):
    if name == "forest_to_houses_1":
        return Curriculum(
            cw = [
                get_named_envconfig("forest_foraging_easy_1"),
                get_named_envconfig("houses_only_easy_1"),
            ],
            cl = [
                300, # length of chapter 1 in episodes
                300, # length of chapter 2 in episodes
                # 100, # length of chapter 1 in episodes
                # 200, # length of chapter 2 in episodes
            ]
        )
    else:
        raise ValueError(f"Unknown curriculum name: {name}")
