
def get_curriculum(name):
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

    if name == "forest_forager_easy_1":
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
    elif name == "forest_foraging_abundant_2":
        pass
    else:
        raise ValueError(f"Unknown curriculum name: {name}")

    return worldgen_config, sensor_config, action_config, reward_config

