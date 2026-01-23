from ratsim.roslike_unity_connector.connector import RoslikeUnityConnector

print("import done")

from ratsim.roslike_unity_connector.connector import *
from ratsim.roslike_unity_connector.message_definitions import *


# class WildfireWorldGenMessage(Message):
#     def __init__(self, seed: int = None, numAgents: int = None, startAndGoalClearingDistance: float = None, arenaWidth: int = None, arenaHeight: int = None, treeDensity: float = None, topology: str = None, treesSwayingFactor: float = None, debrisTriggerzoneSpawnFrequency: float = None, debrisGroupSizeModifier: float = None, carRoadSpawnFrequency: float = None, carVelocityMin: float = None, carVelocityMax: float = None, fireSpawnFrequency: float = None, fireGlobalSpreadModifier: float = None, fireSmokeGenerationModifier: float = None, fireSpreadsAcrossGround: bool = None, staticWindXVel: float = None, staticWindYVel: float = None, windFluctuationModifier: float = None):
#         self.seed = seed
#         self.numAgents = numAgents
#         self.startAndGoalClearingDistance = startAndGoalClearingDistance
#         self.arenaWidth = arenaWidth
#         self.arenaHeight = arenaHeight
#         self.treeDensity = treeDensity
#         self.topology = topology
#         self.treesSwayingFactor = treesSwayingFactor
#         self.debrisTriggerzoneSpawnFrequency = debrisTriggerzoneSpawnFrequency
#         self.debrisGroupSizeModifier = debrisGroupSizeModifier
#         self.carRoadSpawnFrequency = carRoadSpawnFrequency
#         self.carVelocityMin = carVelocityMin
#         self.carVelocityMax = carVelocityMax
#         self.fireSpawnFrequency = fireSpawnFrequency
#         self.fireGlobalSpreadModifier = fireGlobalSpreadModifier
#         self.fireSmokeGenerationModifier = fireSmokeGenerationModifier
#         self.fireSpreadsAcrossGround = fireSpreadsAcrossGround
#         self.staticWindXVel = staticWindXVel
#         self.staticWindYVel = staticWindYVel
#         self.windFluctuationModifier = windFluctuationModifier

if __name__ == "__main__":
    conn = RoslikeUnityConnector()
    conn.connect()

    # Switch to wildfire scene
    scene_name = "Wildfire"
    print(f"Selecting scene: {scene_name}")
    msg = StringMessage(data = scene_name)
    conn.publish(msg, "/sim_control/scene_select")
    conn.send_messages_and_step(enable_physics_step=False)
    rcv = conn.read_messages_from_unity()
    print(f"Received messages after scene select: {rcv}")

    # Prepare and send worldgen message

    # put SOME value to every field in the message
    msg = WildfireWorldGenMessage()
    msg.seed = 42
    msg.numAgents = 1
    msg.startAndGoalClearingDistance = 5.0
    msg.arenaWidth = 1000
    msg.arenaHeight = 1000
    msg.treeDensity = 0.003
    msg.topology = "forest"
    msg.treesSwayingFactor = 1.0
    msg.debrisTriggerzoneSpawnFrequency = 0.1
    msg.debrisGroupSizeModifier = 1.0
    msg.carRoadSpawnFrequency = 0.05
    msg.carVelocityMin = 10.0
    msg.carVelocityMax = 20.0
    msg.fireSpawnFrequency = 0.02
    msg.fireGlobalSpreadModifier = 1.0
    msg.fireSmokeGenerationModifier = 1.0
    msg.fireSpreadsAcrossGround = True
    msg.staticWindXVel = 5.0
    msg.staticWindYVel = 0.0
    msg.windFluctuationModifier = 1.0
    print(f"Sending worldgen message: {msg}")

    conn.publish(msg, "/wildfire_worldgen_input")
    conn.send_messages_and_step(enable_physics_step=False)
    rcv = conn.read_messages_from_unity()
    print(f"Received messages after scene select: {rcv}")


    conn.test_send_and_receive()

