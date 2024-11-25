import carla
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
import time


class CarlaEnv(gym.Env):
    def __init__(self, host="localhost", port=2000):
        super().__init__()

        # CARLA 연결 설정
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        # 관측 공간 정의
        self.observation_space = spaces.Dict(
            {
                "camera": spaces.Box(
                    low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
                ),
                "speed": spaces.Box(low=0, high=50, shape=(1,), dtype=np.float32),
                "location": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(3,), dtype=np.float32
                ),
            }
        )

        # 행동 공간 정의 (조향, 가속, 제동)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # 센서 및 차량 객체 초기화
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.current_image = None
        self.last_collision = False

        # 목표 위치
        self.target_location = None

    def reset(self, start_location=None):
        # 기존 액터 제거
        self._clear_actors()

        # 차량 스폰
        if start_location is None:
            spawn_points = self.map.get_spawn_points()
            start_location = random.choice(spawn_points)

        vehicle_bp = self.blueprint_library.find("vehicle.tesla.model3")
        self.vehicle = self.world.spawn_actor(vehicle_bp, start_location)

        # 카메라 설정
        camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "84")
        camera_bp.set_attribute("image_size_y", "84")
        camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        self.camera.listen(lambda image: self._process_image(image))

        # 충돌 센서 설정
        collision_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        # 초기 관측 반환
        time.sleep(0.1)  # 센서 데이터 수신 대기
        return self._get_obs(), {}

    def step(self, action):
        # 행동 실행
        steer, throttle, brake = action
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )
        self.vehicle.apply_control(control)

        # 다음 상태 관측
        time.sleep(0.1)  # 시뮬레이션 스텝
        obs = self._get_obs()

        # 보상 계산
        reward, done, info = self._compute_reward(obs)

        return obs, reward, done, False, info

    def _get_obs(self):
        location = self.vehicle.get_location()
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        return {
            "camera": self.current_image,
            "speed": np.array([speed], dtype=np.float32),
            "location": np.array(
                [location.x, location.y, location.z], dtype=np.float32
            ),
        }

    def _process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        self.current_image = array

    def _on_collision(self, event):
        self.last_collision = True

    def _compute_reward(self, obs):
        reward = 0
        done = False
        info = {"collision": self.last_collision}

        # 속도에 대한 보상
        speed = obs["speed"][0]
        reward += speed * 0.1

        # 목표 지점까지의 거리에 대한 보상
        if self.target_location is not None:
            current_location = obs["location"]
            distance = np.linalg.norm(current_location - self.target_location)
            reward -= distance * 0.01

            # 목표 도달 확인
            if distance < 2.0:
                reward += 100
                done = True

        # 충돌 페널티
        if self.last_collision:
            reward -= 100
            done = True
            self.last_collision = False

        # 차선 이탈 확인
        waypoint = self.map.get_waypoint(self.vehicle.get_location())
        if not waypoint.lane_type == carla.LaneType.Driving:
            reward -= 50
            done = True

        return reward, done, info

    def _clear_actors(self):
        if self.camera is not None:
            self.camera.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        if self.vehicle is not None:
            self.vehicle.destroy()
