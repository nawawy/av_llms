import time
from carla_client import CarlaClient
from perception import cluster_point_cloud, extract_object_info
from rule_detector import rule_based_detector
from fsm import DrivingFSM
from config import LOOP_SLEEP


def run_once(duration=20.0):
    client = CarlaClient()
    try:
        vehicle = client.spawn_ego()
        client.attach_lidar()
        fsm = DrivingFSM()
        t0 = time.time()
        while time.time() - t0 < duration:
            data = client.get_latest_lidar()
            if data is None:
                time.sleep(0.01)
                continue
            pts = data['points']
            clusters = cluster_point_cloud(pts)
            objects = [extract_object_info(c) for c in clusters]
            ego_speed = client.get_ego_speed()
            # NOTE: relative_speed estimation placeholder -> 0.0, you should implement tracker
            hazard = rule_based_detector(ego_speed, objects)
            state = fsm.update(hazard)
            control = fsm.get_controls()
            client.apply_control(control['throttle'], control['brake'], control['steer'])
            time.sleep(LOOP_SLEEP)
    except KeyboardInterrupt:
        pass
    finally:
        client.destroy()

if __name__ == '__main__':
    run_once(30.0)