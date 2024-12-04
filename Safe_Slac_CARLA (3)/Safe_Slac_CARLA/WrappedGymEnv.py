import gym
from gym.spaces.box import Box
import numpy as np
from PIL import Image
import ray


class WrappedGymEnv(gym.Wrapper):
    def __init__(self, env, **kargs):
        super(WrappedGymEnv, self).__init__(env)

        self.height = kargs['image_size']
        self.width = kargs['image_size']
        self.action_repeat = kargs['action_repeat']
        self._max_episode_steps = 1000

        # 사용하는 센서따라 채널 변경 될것 같다 -- 설정 필요
        """
        현재 고려하는 센서 
        좌우 앞 위 카메라 --3*4= 12
        라이다 레이더 -- 3*2 = 6

        18 채널 고려
        """
        self.observation_space = Box(0, 255, (18, self.height, self.width), np.uint8)

        self.ometer_space = Box(-np.inf, np.inf, shape=(40, 2), dtype=np.float32)
        self.tgt_state_space = Box(0, 255, (3, self.height, self.width), np.uint8)

        self.action_space = Box(-1.0, 1.0, shape=(2,))
        self.env = env

    def get_env_shape(self):
        return {
            "env_observation_space_shape": self.observation_space.shape,
            "env_ometer_space_shape": self.ometer_space.shape,
            "env_tgt_state_space_shape": self.tgt_state_space.shape,
            "env_action_space_shape": self.action_space.shape
        }

    def reset(self):
        # reset_output_ref = self.env.reset.remote()#self.env.reset()
        # reset_output = ray.get(reset_output_ref)

        reset_output = self.env.reset()

        img_np = reset_output['left_camera']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_1 = np.transpose(img_np_resized, [2, 1, 0])

        img_np = reset_output['front_camera']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_2 = np.transpose(img_np_resized, [2, 1, 0])

        img_np = reset_output['right_camera']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_3 = np.transpose(img_np_resized, [2, 1, 0])

        img_np = reset_output['top_camera']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_4 = np.transpose(img_np_resized, [2, 1, 0])

        img_np = reset_output['lidar_image']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_5 = np.transpose(img_np_resized, [2, 1, 0])

        img_np = reset_output['radar_image']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_6 = np.transpose(img_np_resized, [2, 1, 0])

        src_img = np.concatenate((src_img_1,
                                  src_img_2,
                                  src_img_3,
                                  src_img_4,
                                  src_img_5,
                                  src_img_6), axis=0)

        img_np = reset_output['hud']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        tgt_img = np.transpose(img_np_resized, [2, 0, 1])

        wpsh = reset_output['wp_hrz']

        return src_img, wpsh, tgt_img

    def step(self, action):
        if action[0] > 0:
            throttle = np.clip(action[0], 0.0, 1.0)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-action[0], 0.0, 1.0)

        act_tuple = ([throttle, brake, action[1]], [False])

        for _ in range(self.action_repeat):
            re_ref = self.env.step(action)  # .remote(act_tuple)

        re = re_ref  # ray.get(re_ref)

        re = list(re)
        img_np = re[0]['left_camera']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_1 = np.transpose(img_np_resized, [2, 1, 0])

        img_np = re[0]['front_camera']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_2 = np.transpose(img_np_resized, [2, 1, 0])

        img_np = re[0]['right_camera']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_3 = np.transpose(img_np_resized, [2, 1, 0])

        img_np = re[0]['top_camera']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_4 = np.transpose(img_np_resized, [2, 1, 0])

        img_np = re[0]['lidar_image']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_5 = np.transpose(img_np_resized, [2, 1, 0])

        img_np = re[0]['radar_image']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_6 = np.transpose(img_np_resized, [2, 1, 0])

        src_img = np.concatenate((src_img_1,
                                  src_img_2,
                                  src_img_3,
                                  src_img_4,
                                  src_img_5,
                                  src_img_6), axis=0)

        img_np = re[0]['hud']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height, self.width))
        img_np_resized = np.uint8(img_pil_resized)
        tgt_img = np.transpose(img_np_resized, [2, 0, 1])

        wpsh = re[0]['wp_hrz']

        return src_img, wpsh, tgt_img, re[1], re[2], re[3]  # src_img,odometer,target image , ,reward, done, info(cost)

    def pid_sample(self):
        # pid_ref = self.env.pid_sample.remote()
        # pid_sample = ray.get(pid_ref)
        # return pid_sample #
        return self.env.pid_sample()