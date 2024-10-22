import ray
import numpy as np
import time
import argparse
from SAC_for_carla_v2.sac import SAC
from SAC_for_carla_v2.PER import PER_Buffer
from SAC_for_carla_v2.utils import LinearSchedule, adaptiveSchedule, Experience

FLAG=True# 삭제해야하는 코드


# 통합을 담당하는 Actor 정의
@ray.remote(num_gpus=1)
class ParameterServer:
    def __init__(self, args, device, expected_workers = 2):

        self.batch_size = args.batch_size
        self.start_timesteps = args.start_timesteps

        self.collected_weights = []
        self.expected_workers = expected_workers

        self.buffer = PER_Buffer(args, device=device)
        self.device = device

        self.BETA = args.beta_init
        self.BUFFER_ALPHA = 0.6

        self.beta_scheduler = LinearSchedule(args.beta_gain_steps, args.beta_init, 1.0)
        self.buffer_alpha_scheduler = adaptiveSchedule(args.alpha_max, args.alpha_min)

        self.actor_lr_scheduler = LinearSchedule(args.lr_decay_steps, args.lr_init, args.lr_end)
        self.critic_lr_scheduler = LinearSchedule(args.lr_decay_steps, args.lr_init, args.lr_end)

        # 정책 초기화 (옵션)
        self.policy = SAC(args, device)

        print("ParameterServer initialized successfully.")

    def add_experience(self, experience):
        self.buffer.add(experience)

    def update_hyperparameters(self, t):
        self.BETA = self.beta_scheduler.value(t)
        td_errors = self.buffer.buffer[:len(self.buffer)]["priority"]
        td_mean = np.mean(td_errors)
        td_std = np.std(td_errors)
        self.BUFFER_ALPHA = self.buffer_alpha_scheduler.value(td_mean, td_std)

    def train_policy(self, t):
        if len(self.buffer) > self.batch_size and t > self.start_timesteps:

            print(f"policy 업데이트. 경험 개수: {len(self.buffer)}")
            self.policy.train(self.BETA, self.BUFFER_ALPHA, self.buffer)
            self.update_hyperparameters(t)
         
            # Actor lr scheduler
            for p in self.policy.actor_optimizer.param_groups:
                p['lr'] = self.actor_lr_scheduler.value(t)

            # Critic lr scheduler
            for p in self.policy.critic_optimizer.param_groups:
                p['lr'] = self.critic_lr_scheduler.value(t)

            for p in self.policy.critic_optimizer2.param_groups:
                p['lr'] = self.critic_lr_scheduler.value(t)
    
    def get_policy_weights(self):
        weights = {}
        weights['actor'] = {name: param.detach().cpu().numpy() for name, param in self.policy.actor.named_parameters()}
        weights['critic1'] = {name: param.detach().cpu().numpy() for name, param in self.policy.critic.named_parameters()}
        weights['critic2'] = {name: param.detach().cpu().numpy() for name, param in self.policy.critic2.named_parameters()}
        weights['log_alpha'] = self.policy.log_alpha.detach().cpu().numpy()
        return weights