import gym
from gym.spaces import Box, Dict
import numpy as np
from PIL import Image


class WrappedGymEnv(gym.Wrapper):
    def __init__(self, env, **kargs):
        super(WrappedGymEnv, self).__init__(env)
        self.height = kargs["image_size"]
        self.width = kargs["image_size"]
        self.action_repeat = kargs["action_repeat"]
        self._max_episode_steps = 1000
        self.steps = 0

        self.observation_space = Dict(
            {
                "image": Box(0, 255, (18, self.height, self.width), np.uint8),
                "waypoints": Box(-np.inf, np.inf, (40, 2), np.float32),
                "tgt_image": Box(0, 255, (3, self.height, self.width), np.uint8),
            }
        )

        self.action_space = Box(-1.0, 1.0, shape=(2,))
        self.env = env

    def reset(self):
        self.steps = 0
        raw_obs = self.env.reset()
        obs = self._convert_obs(raw_obs)
        return obs

    def step(self, action):
        self.steps += 1
        if action[0] > 0:
            throttle = np.clip(action[0], 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-action[0], 0.0, 1.0)
        steer = np.clip(action[1], -1.0, 1.0)

        act_tuple = ([throttle, brake, steer], [False])
        done = False
        reward_sum = 0.0
        info = {}
        obs = None
        for _ in range(self.action_repeat):
            raw_obs, r, d, i = self.env.step(act_tuple)
            reward_sum += r
            info = i
            obs = raw_obs
            if d:
                done = True
                break
        obs = self._convert_obs(obs)
        if self.steps >= self._max_episode_steps:
            done = True
        return obs, reward_sum, done, info

    def _convert_obs(self, out):
        left = self._resize_image(out["left_camera"], (3, self.height, self.width))
        front = self._resize_image(out["front_camera"], (3, self.height, self.width))
        right = self._resize_image(out["right_camera"], (3, self.height, self.width))
        top = self._resize_image(out["top_camera"], (3, self.height, self.width))
        lidar = self._resize_image(out["lidar_image"], (3, self.height, self.width))
        radar = self._resize_image(out["radar_image"], (3, self.height, self.width))
        tgt_img = self._resize_image(out["hud"], (3, self.height, self.width))
        src_img = np.concatenate((left, front, right, top, lidar, radar), axis=0)
        wpsh = out["wp_hrz"]
        return {"image": src_img, "waypoints": wpsh, "tgt_image": tgt_img}

    def _resize_image(self, img_np, shape):
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((shape[2], shape[1]))
        img_np_resized = np.uint8(img_pil_resized)
        img_np_resized = np.transpose(img_np_resized, [2, 0, 1])
        return img_np_resized
