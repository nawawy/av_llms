import carla
import time
import threading
import numpy as np
from collections import deque
from config import CARLA_HOST, CARLA_PORT, LIDAR_CHANNELS, LIDAR_RANGE, LIDAR_ROTATION_FREQ, LIDAR_POINTS_PER_SEC

class CarlaClient:
    def __init__(self, host=CARLA_HOST, port=CARLA_PORT):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle = None
        self.lidar_sensor = None
        self.lidar_data_queue = deque(maxlen=20)
        self.lock = threading.Lock()


    def spawn_ego(self, spawn_point=None, vehicle_bp_filter='vehicle.tesla.model3'):
        bp = self.blueprint_library.filter(vehicle_bp_filter)[0]
        if spawn_point is None:
            spawn_point = self.world.get_map().get_spawn_points()[0]
            self.vehicle = self.world.spawn_actor(bp, spawn_point)
            print(f"Spawned ego vehicle: {self.vehicle}")
        return self.vehicle


    def attach_lidar(self):
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', str(LIDAR_CHANNELS))
        lidar_bp.set_attribute('range', str(LIDAR_RANGE))
        lidar_bp.set_attribute('rotation_frequency', str(LIDAR_ROTATION_FREQ))
        lidar_bp.set_attribute('points_per_second', str(int(LIDAR_POINTS_PER_SEC)))


        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
        self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        self.lidar_sensor.listen(lambda data: self._lidar_callback(data))
        print("LiDAR attached and listening")


    def _lidar_callback(self, point_cloud):
        points = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        with self.lock:
            self.lidar_data_queue.append({'timestamp': time.time(), 'points': points})


    def get_latest_lidar(self):
        with self.lock:
            return self.lidar_data_queue[-1] if len(self.lidar_data_queue) > 0 else None


    def get_ego_speed(self):
        if self.vehicle is None:
            return 0.0
        vel = self.vehicle.get_velocity()
        return (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5


    def apply_control(self, throttle, brake, steer):
        from carla import VehicleControl
        vc = VehicleControl()
        vc.throttle = float(max(0.0, min(1.0, throttle)))
        vc.brake = float(max(0.0, min(1.0, brake)))
        vc.steer = float(max(-1.0, min(1.0, steer)))
        self.vehicle.apply_control(vc)


    def destroy(self):
        actors = [self.lidar_sensor, self.vehicle]
        for a in actors:
            if a is not None:
                try:
                    a.destroy()
                except Exception:
                    pass