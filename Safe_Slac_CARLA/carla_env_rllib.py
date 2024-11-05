import gym
from gym.spaces import Box, Dict, Discrete
import numpy as np
import ray
from ray.rllib.env.env_context import EnvContext


class CarlaEnvRllib(gym.Env):
    def __init__(self, config: EnvContext):
        # 기존 CarlaRlEnv 파라미터 설정 가져오기
        params = {
            "carla_port": config.get("carla_port", 2000),
            "map_name": config.get("map_name", "Town10HD"),
            "window_resolution": config.get("window_resolution", [1080, 1080]),
            "grid_size": config.get("grid_size", [3, 3]),
            "sync": config.get("sync", True),
            "no_render": config.get("no_render", True),
            "display_sensor": config.get("display_sensor", False),
            "ego_filter": config.get("ego_filter", "vehicle.tesla.model3"),
            "num_vehicles": config.get("num_vehicles", 50),
            "num_pedestrians": config.get("num_pedestrians", 20),
            "enable_route_planner": config.get("enable_route_planner", True),
            "sensors_to_amount": config.get(
                "sensors_to_amount",
                ["left_rgb", "front_rgb", "right_rgb", "top_rgb", "lidar", "radar"],
            ),
        }

        # 기존 CarlaRlEnv 초기화
        self.env = gym.make("CarlaRlEnv-v0", params=params)

        # observation space 정의
        self.observation_space = Dict(
            {
                "camera": Box(0, 255, shape=(64, 64, 18), dtype=np.uint8),
                "measurements": Box(-np.inf, np.inf, shape=(40, 2), dtype=np.float32),
                "target": Box(0, 255, shape=(64, 64, 3), dtype=np.uint8),
            }
        )

        # action space 정의
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self):
        state = self.env.reset()
        return self._process_obs(state)

    def step(self, action):
        # action을 CarlaRlEnv 형식으로 변환
        carla_action = (
            [action[0], 0.0, action[1]],  # throttle, brake, steer
            [0],  # reverse flag
        )

        obs, reward, done, info = self.env.step(carla_action)

        return self._process_obs(obs), reward, done, info

    def _process_obs(self, obs):
        # observation을 RLlib에 맞게 처리
        processed_obs = {
            "camera": np.stack(
                [
                    obs["front_camera"],
                    obs["left_camera"],
                    obs["right_camera"],
                    obs["top_camera"],
                    obs["lidar_image"],
                    obs["radar_image"],
                ],
                axis=-1,
            ),
            "measurements": obs["wp_hrz"],
            "target": obs["hud"],
        }
        return processed_obs

    def render(self):
        self.env.display()
