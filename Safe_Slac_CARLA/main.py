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
from central import ParameterServer
from gym.envs.registration import register

FLAG=True# 삭제해야하는 코드

@ray.remote
class Worker:
    def __init__(self, worker_id, _args, port, traffic_port):

        self.args = _args
        self.config = get_default_config()
        self.config["domain_name"] = self.args.domain_name
        self.config["task_name"] = self.args.task_name
        self.config["seed"] = self.args.seed
        self.config["num_steps"] = self.args.num_steps

        self.central_server = ray.get_actor("ParameterServer", namespace="parameter_server_namespace")

        # env params
        self.params = {
            'carla_port': port,
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
            'sensors_to_amount': ['left_rgb','front_rgb', 'right_rgb','top_rgb','lidar','radar'],
        }

        self.env = WrappedGymEnv(gym.make(self.args.domain_name, params=self.params),
                            action_repeat=self.args.action_repeat,image_size=64)
        self.env_test = self.env


        self.log_dir = os.path.join(
            "logs",
            f"{self.config['domain_name']}-{self.config['task_name']}",
            f'slac-seed{self.config["seed"]}-{datetime.now().strftime("%Y%m%d-%H%M")}',
        )
        self.algo = LatentPolicySafetyCriticSlac(
            num_sequences=self.config["num_sequences"],
            gamma_c=self.config["gamma_c"],
            state_shape=self.env.observation_space.shape,
            ometer_shape=self.env.ometer_space.shape,
            tgt_state_shape=self.env.tgt_state_space.shape,
            action_shape=self.env.action_space.shape,
            action_repeat=self.config["action_repeat"],
            device=torch.device("cuda" if self.args.cuda else "cpu"),
            seed=self.config["seed"],
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
            is_worker=True
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

    ray.shutdown()
    ray.init(namespace="parameter_server_namespace")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    register(
        id='CarlaRlEnv-v0',
        entry_point='carla_rl_env.carla_env:CarlaRlEnv',
        max_episode_steps=1000,
    )
    temp_env = WrappedGymEnv(gym.make("CarlaRlEnv-v0", params=params), args)
    args.action_shape = temp_env.action_space.shape[0]
    args.action_scale = temp_env.action_scale
    args.action_bias = temp_env.action_bias

    parameter_server = ParameterServer.options(
        name="ParameterServer",
        namespace="parameter_server_namespace",  # 네임스페이스 추가
        lifetime="detached"
    ).remote(args, device, expected_workers=num_workers)

    workers = []
    for worker_id in range(num_workers):
        port = args.port + worker_id * 100
        print(port)
        traffic_port = args.traffic_port + worker_id * 100
        worker = Worker.remote(worker_id, args, port, traffic_port)
        workers.append(worker)

    # 각 워커에서 학습 시작
    train_ids = [worker.train.remote(parameter_server) for worker in workers]

    print("Central server is ready to collect parameters...")
    print("Current actors:", ray.util.list_named_actors())


if __name__ == '__main__':
    main()