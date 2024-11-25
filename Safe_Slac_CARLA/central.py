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
import carla_rl_env
from configuration import get_default_config


# 통합을 담당하는 Actor 정의
@ray.remote(num_gpus=1)
class CentralServer:
    def __init__(self, args, expected_workers=2):
        self.args = args
        self.expected_workers = expected_workers
        
        # 환경 설정
        self.env, self.env_test = self.setup_environment()
        
        # 로그 디렉토리 설정
        self.log_dir = self.setup_log_directory()

        # SLAC 알고리즘 초기화
        self.algo = self.initialize_slac_algorithm()

        # 버퍼 초기화
        self.buffer = self.algo.buffer

        self.trainer = self.initialize_trainer()
        self.trainer.writer.add_text("config", json.dumps(vars(args)), 0) #텐서보드 시각화 위한 코드


        print("ParameterServer initialized successfully.")

    def setup_environment(self):
        # 칼라환경 파라미터 정의
        params = {
            'carla_port': 2000,
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

    def initialize_slac_algorithm(self, config): #알고리즘 초기화, 상태 및 행동공간 형태도 지정
        return LatentPolicySafetyCriticSlac(
            state_shape=self.env.observation_space.shape,
            ometer_shape=self.env.ometer_space.shape,
            tgt_state_shape=self.env.tgt_state_space.shape,
            action_shape=self.env.action_space.shape,
            action_repeat=config["action_repeat"],
            device=torch.device("cuda" if config["cuda"] else "cpu"),
            seed=config["seed"],
            buffer_size=config["buffer_size"],
            num_sequences=config["num_sequences"],
            feature_dim=config["feature_dim"],
            z1_dim=config["z1_dim"],
            z2_dim=config["z2_dim"],
            hidden_units=config["hidden_units"],
            batch_size_latent=config["batch_size_latent"],
            batch_size_sac=config["batch_size_sac"],
            lr_sac=config["lr_sac"],
            lr_latent=config["lr_latent"],
            start_alpha=config["start_alpha"],
            start_lagrange=config["start_lagrange"],
            grad_clip_norm=config["grad_clip_norm"],
            tau=config["tau"],
            image_noise=config["image_noise"],
        )

    def load_model(self): #사전 훈련 모델 로드
        self.algo.load_model("logs/tmp")

    def initialize_trainer(self):
        trainer = Trainer(
        num_sequences=config["num_sequences"],
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        seed=config["seed"],
        num_steps=config["num_steps"],
        initial_learning_steps=config["initial_learning_steps"],
        initial_collection_steps=config["initial_collection_steps"],
        collect_with_policy=config["collect_with_policy"],
        eval_interval=config["eval_interval"],
        num_eval_episodes=config["num_eval_episodes"],
        action_repeat=config["action_repeat"],
        train_steps_per_iter=config["train_steps_per_iter"],
        env_steps_per_train_step=config["env_steps_per_train_step"]
        )
        return trainer

    def add_experience(self, experience): # 워커의 experience가 저장, woker에서 실행되는 코드
        self.buffer.add(experience)

    def train(self):
        trainer = self.initialize_trainer()
        while True:
            if self.buffer_has_data():
                trainer.train()
            time.sleep(5)
        #학습 시에 버퍼의 데이터는 어떻게 처리할지

    def buffer_has_data(self):
        return len(self.buffer) > self.args.batch_size
    
    #워커에서 요청할 때 센트럴에서 함수가 필요한지 질문
    def update_worker(self):
        updated_parameters = self.algo.get_parameters()

        return updated_parameters

def main():
    config = get_default_config()
    '''config["domain_name"] = args.domain_name
    config["task_name"] = args.task_name
    config["seed"] = args.seed
    config["num_steps"] = args.num_steps'''

    ray.init()

    central = CentralServer.remote(config)
    # 환경 초기화
    env, env_test = ray.get(central.setup_environment.remote())
    print("Environment initialized.")
                  
    # 트레이너 생성
    trainer = ray.get(central.initialize_trainer.remote(config))
    print("Trainer initialized.")

    # 학습 시작
    # 이 코드는 중앙 서버가 학습을 시작하도록 합니다.
    ray.get(central.train.remote())
    print("Training started.")
    

