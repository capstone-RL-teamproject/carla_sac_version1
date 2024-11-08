# 초기 버퍼 생성
# 워커에서 매 experience 마다 add 해서 버퍼업데
# 센트럴은 5초마다 버퍼를 통해 학습
# 워커에서 일정 스텝만큼 진행하면 센트럴로부터 파라미터 가져와서 업데이트

import ray
import numpy as np
import time
import argparse
from safe_slace.algo import LatentPolicySafetyCriticSlac
import torch
import os
from datetime import datetime

from WrappedGymEnv import WrappedGymEnv
import gym
from safe_slace.trainer import Trainer
import json
from configuration import get_default_config
from gym.envs.registration import register
import carla_rl_env


# 통합을 담당하는 Actor 정의
@ray.remote(num_gpus=1)
class CentralServer:
    def __init__(self, args, device, expected_workers=2):
        self.args = args
        self.config = get_default_config()
        #print(self.config)
        self.expected_workers = expected_workers
        self.is_ready = True

        register(
            id='CarlaRlEnv-v0',
            entry_point='carla_rl_env.carla_env:CarlaRlEnv',
            max_episode_steps=1000,
        )
        
        # 환경 설정
        self.env, self.env_test = self.setup_environment()
        
        # 로그 디렉토리 설정
        self.log_dir = self.setup_log_directory()

        # SLAC 알고리즘 초기화
        self.algo = self.initialize_slac_algorithm()

        self.trainer = self.initialize_trainer()
        self.trainer.writer.add_text("config", json.dumps(vars(args)), 0) #텐서보드 시각화 위한 코드


        print("ParameterServer initialized successfully.")

    def setup_environment(self):
        #칼라환경 파라미터 정의
        self.config["domain_name"] = self.args.domain_name
        self.config["task_name"] = self.args.task_name
        self.config["seed"] = self.args.seed
        self.config["num_steps"] = self.args.num_steps

        params = {
            'carla_port': 3000,
            'map_name': 'Town10HD',
            'window_resolution': [1080, 1080],
            'grid_size': [3, 3],
            'sync': True,
            'no_render': False,
            'display_sensor': True,
            'ego_filter': 'vehicle.tesla.model3',
            'num_vehicles': 50,
            'num_pedestrians': 20,
            'enable_route_planner': True,
            'sensors_to_amount': ['left_rgb', 'front_rgb', 'right_rgb', 'top_rgb', 'lidar', 'radar'],
        }

        # 환경 설정
        env = WrappedGymEnv(gym.make(self.args.domain_name, params=params), 
                             action_repeat=self.args.action_repeat, 
                             image_size=64)
        env_test = env  # 테스트 환경을 동일하게 설정
        
        return env, env_test

    def setup_log_directory(self): #로그 저장 디렉토리 설정
        return os.path.join(
            "logs",
            f"{self.args.domain_name}-{self.args.task_name}",
            f'slac-seed{self.args.seed}-{datetime.now().strftime("%Y%m%d-%H%M")}',
        )

    def initialize_slac_algorithm(self): #알고리즘 초기화, 상태 및 행동공간 형태도 지정
        return LatentPolicySafetyCriticSlac(
            num_sequences=self.config["num_sequences"],
            gamma_c=self.config["gamma_c"],
            state_shape=self.env.observation_space.shape,
            ometer_shape=self.env.ometer_space.shape,
            tgt_state_shape=self.env.tgt_state_space.shape,
            action_shape=self.env.action_space.shape,
            action_repeat=self.config["action_repeat"],
            device=torch.device("cuda" if self.args.cuda else "cpu"),
            seed=self.config["seed"],
            parameter_server=None,
            buffer_size=self.config["buffer_size"],
            feature_dim=self.config["feature_dim"],
            z2_dim=self.config["z2_dim"],
            hidden_units=self.config["hidden_units"],
            batch_size_latent=self.config["batch_size_latent"],
            batch_size_sac=self.config["batch_size_sac"],
            lr_sac=self.config["lr_sac"],
            lr_latent=self.config["lr_latent"],
            start_alpha=self.config["start_alpha"],
            start_lagrange=self.config["start_lagrange"],
            grad_clip_norm=self.config["grad_clip_norm"],
            tau=self.config["tau"],
            image_noise=self.config["image_noise"],
        )

    def load_model(self): #사전 훈련 모델 로드
        self.algo.load_model("logs/tmp")

    def initialize_trainer(self, is_worker = False):
        trainer = Trainer(
            num_sequences=self.config["num_sequences"],
            env=self.env,
            env_test=self.env_test,
            algo=self.algo,
            log_dir=self.log_dir,
            parameter_server=None,
            seed=self.config["seed"],
            num_steps=self.config["num_steps"],
            initial_learning_steps=self.config["initial_learning_steps"],
            initial_collection_steps=self.config["initial_collection_steps"],
            collect_with_policy=self.config["collect_with_policy"],
            eval_interval=self.config["eval_interval"],
            num_eval_episodes=self.config["num_eval_episodes"],
            action_repeat=self.config["action_repeat"],
            train_steps_per_iter=self.config["train_steps_per_iter"],
            env_steps_per_train_step=self.config["env_steps_per_train_step"],
            is_worker = False
        )
        return trainer

    def add_buffer(self, action, reward, mask, state, ometer, tgt_state, done, cost): # 워커의 experience가 저장, woker에서 실행되는 코드
        self.algo.buffer.append(action, reward, mask, state, ometer, tgt_state, done, cost)
    
    #워커에서 요청할 때 센트럴에서 함수가 필요한지 질문
    def update_worker(self):
        updated_parameters = self.algo.get_parameters()

        return updated_parameters
    
    def get_buffer_size(self):
        return len(self.buffer)
    
    def train(self):
        self.trainer.train()

    def ready(self):
        return self.is_ready