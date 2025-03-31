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





        # 이미지 확인하기
        # print('\n--------------------\n')

        # # src_img: first 3 channels
        # src_img_1_display = np.transpose(src_img[:3], [2, 1, 0])
        #
        # # src_img: last 3 channels
        # src_img_2_display = np.transpose(src_img[3:], [2, 1, 0])
        #
        # # tgt_img: target image (transposed back for visualization)
        # tgt_img_display = np.transpose(tgt_img, [1, 2, 0])
        #
        # # Plotting src_img (split into 2) and tgt_img
        # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        #
        # # Displaying the first 3 channels of src_img
        # axs[0].imshow(src_img_1_display)
        # axs[0].set_title('src_img (First 3 channels)')
        # axs[0].axis('off')
        #
        # # Displaying the last 3 channels of src_img
        # axs[1].imshow(src_img_2_display)
        # axs[1].set_title('src_img (Last 3 channels)')
        # axs[1].axis('off')
        #
        # # Displaying the tgt_img
        # axs[2].imshow(tgt_img_display)
        # axs[2].set_title('tgt_img')
        # axs[2].axis('off')
        #
        # # Show the plot
        # plt.show()
        # print('\n--------------------\n')
        #print(f"\n\nwrapper gym env R :{re[1]}  D :{re[2]}  I :{re[3]}\n\n")
        return tgt_img, re[1], re[2], re[3]  # src_img,wpsh,~~~~  observation ,reward, done, info
