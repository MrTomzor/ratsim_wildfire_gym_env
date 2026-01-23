from ratsim.roslike_unity_connector.connector import RoslikeUnityConnector

print("import done")

from ratsim.roslike_unity_connector.connector import *
from ratsim.roslike_unity_connector.message_definitions import *

if __name__ == "__main__":
    conn = RoslikeUnityConnector()
    conn.connect()

    scene_name = "BoxArena"
    print(f"Selecting scene: {scene_name}")
    msg = StringMessage(data = scene_name)
    conn.publish(msg, "/sim_control/scene_select")
    conn.send_messages_and_step(enable_physics_step=False)
    rcv = conn.read_messages_from_unity()
    print(f"Received messages after scene select: {rcv}")

    conn.test_send_and_receive()

