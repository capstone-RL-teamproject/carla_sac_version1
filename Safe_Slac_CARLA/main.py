import argparse
import os
from datetime import datetime
import random

from matplotlib.pyplot import get

import torch

from safe_slace.algo import LatentPolicySafetyCriticSlac
from torch.utils.tensorboard import SummaryWriter
from WrappedGymEnv import WrappedGymEnv
import argparse
import time

import gym
from safe_slace.trainer import Trainer
import json
from configuration import get_default_config

import ray
from gym.envs.registration import register
from central import CentralServer

FLAG=True# 삭제해야하는 코드

@ray.remote
class Worker:
    def __init__(self, worker_id, _args, port, traffic_port, central_server):

        self.args = _args
        self.config = get_default_config()
        self.config["domain_name"] = self.args.domain_name
        self.config["task_name"] = self.args.task_name
        self.config["seed"] = self.args.seed
        self.config["num_steps"] = self.args.num_steps

        self.central_server = central_server
        # self.env, _ = ray.get(self.central_server.setup_environment.remote())
        self.weights = ray.get(self.central_server.update_worker.remote())
        self.algo = ray.get(self.central_server.initialize_slac_algorithm.remote())
        self.trainer = ray.get(self.central_server.initialize_trainer.remote(is_worker=True))
        
        params = {
            'carla_port': self.args.port,
            'traffic_port': traffic_port,
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
        self.env = WrappedGymEnv(gym.make(self.args.domain_name, params=params), 
                             action_repeat=self.args.action_repeat, 
                             image_size=64) #traffic port 때문에 central에서 안받아옴
        
        self.log_dir = os.path.join(
            "logs",
            f"{self.config['domain_name']}-{self.config['task_name']}",
            f'slac-seed{self.config["seed"]}-{datetime.now().strftime("%Y%m%d-%H%M")}',
        )

        self.algo.load_model("logs/tmp")

        self.trainer = Trainer(
            num_sequences=self.config["num_sequences"],
            env=self.env,
            env_test=self.env_test,
            algo=self.algo,
            log_dir=self.log_dir,
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
        )

    def update_policy(self, new_weights):
        # Actor 업데이트
        for name, param in self.algo.actor.named_parameters():
            param.data.copy_(torch.from_numpy(new_weights['actor'][name]))
        # Critic1 업데이트
        for name, param in self.algo.critic.named_parameters():
            param.data.copy_(torch.from_numpy(new_weights['critic1'][name]))
        # Critic Target 업데이트
        for name, param in self.algo.critic_target.named_parameters():
            param.data.copy_(torch.from_numpy(new_weights['critic_target'][name]))
        # Safety Critic 업데이트
        for name, param in self.algo.safety_critic.named_parameters():
            param.data.copy_(torch.from_numpy(new_weights['safety_critic'][name]))
        # Safety Critic Target 업데이트
        for name, param in self.algo.safety_critic.named_parameters():
            param.data.copy_(torch.from_numpy(new_weights['safety_critic_target'][name]))
        # Latent 업데이트
        for name, param in self.algo.latent.named_parameters():
            param.data.copy_(torch.from_numpy(new_weights['latent'][name]))
        
    def train(self, parameter_server):
        self.trainer.train(parameter_server, use_update=False)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=2 * 10 ** 6)
    parser.add_argument("--domain_name", type=str, default="CarlaRlEnv-v0")
    parser.add_argument("--task_name", type=str, default="run")
    parser.add_argument("--action_repeat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_false")

    parser.add_argument('--port', type=int, default=3000, help='Port num')
    parser.add_argument('--traffic_port', type=int, default=8000, help='Traffic port num')
    parser.add_argument('--num_workers', type=int, default=2, help='Workers num')

    args = parser.parse_args()

    num_workers = args.num_workers

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = get_default_config()  # 기본 설정값 불러오기
    
    # Ray 초기화
    ray.init(log_to_driver=True, logging_level="DEBUG")

    # Central Server 초기화
    central = CentralServer.options(
        name="CentralServer",
        namespace="central_namespace",
        lifetime="detached"  
    ).remote(args, device) 
    print("Central server initialized.")
    print(central)
    # 환경 초기화
    # env, env_test = central.setup_environment.remote()
    # print("Environment initialized.")


    workers = []
    for worker_id in range(num_workers):
        port = args.port + worker_id * 100
        print(port)
        traffic_port = args.traffic_port + worker_id * 100
        worker = Worker.remote(worker_id, args, port, traffic_port, central)
        workers.append(worker)
    print(workers)

    # 각 워커에서 학습 시작
    train_ids = [worker.train.remote(central) for worker in workers]

    print("Central server is ready to collect parameters...")
    print("Current actors:", ray.util.list_named_actors())

    while True:
        central.train.remote()


if __name__ == '__main__':
    main()