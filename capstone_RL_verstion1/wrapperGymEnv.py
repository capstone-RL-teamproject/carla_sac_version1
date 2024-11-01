import gym
from gym.spaces.box import Box
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class WrappedGymEnv(gym.Wrapper):
    def __init__(self,env,opt):
        super(WrappedGymEnv,self).__init__(env)
        self.task_name=opt.task_name
        self.height=opt.image_size
        self.width=opt.image_size
        self.action_repeat=opt.action_repeat
        self._max_episode_steps = 1000
        self.observation_space=Box(0, 255, (6,self.height,self.width), np.uint8)
        self.ometer_space=Box(-np.inf, np.inf, shape=(40,2), dtype=np.float32)
        self.tgt_state_space=Box(0, 255, (3,self.height,self.width), np.uint8)


        self.action_space = Box(-1.0, 1.0, shape=(2,))
        self.action_scale = (self.action_space.high - self.action_space.low)/2.0
        self.action_bias = (self.action_space.high + self.action_space.low)/2.0

        self.env=env

    def reset(self):

        reset_output = self.env.reset()

        img_np = reset_output['hud']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height,self.width))
        img_np_resized = np.uint8(img_pil_resized)
        tgt_img = np.transpose(img_np_resized,[2,0,1])



        return tgt_img

    def step(self, action):
        if action[0] > 0:
            throttle = np.clip(action[0],0.0,1.0)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-action[0],0.0,1.0)
        act_tuple = ([throttle, brake, action[1]],[False]) # Tuple(Box,Discret)

        for _ in range(self.action_repeat):
            re = self.env.step(act_tuple)

        re=list(re)


        img_np = re[0]['hud']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height,self.width))
        img_np_resized = np.uint8(img_pil_resized)
        tgt_img = np.transpose(img_np_resized,[2,0,1])




        return tgt_img, re[1], re[2], re[3]  # src_img,wpsh,~~~~  observation ,reward, done, info
