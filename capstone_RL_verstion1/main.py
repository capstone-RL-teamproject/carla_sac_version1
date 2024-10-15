from torch.utils.tensorboard import SummaryWriter
from wrapperGymEnv import *
import argparse
import torch
import os

from carla_rl_env.carla_env import CarlaRlEnv
from SAC_for_carla_v2.sac import SAC
from SAC_for_carla_v2.PER import PER_Buffer
from SAC_for_carla_v2.utils import *
import gym
import ray

FLAG=True# 삭제해야하는 코드

@ray.remote
class Worker:
    def __init__(self, worker_id, _params, _args, file_name, port, traffic_port):
        from gym.envs.registration import register
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("\nDevice is ", device)
        print()

        register(
            id='CarlaRlEnv-v0',
            entry_point='carla_rl_env.carla_env:CarlaRlEnv',
            max_episode_steps=1000,
        )

        self.id = worker_id

        self.args = _args
        self.params = _params
        self.args.port = port
        self.args.traffic_port = traffic_port
        self.params['carla_port'] = port
        self.params['traffic_port'] = traffic_port

        self.env = WrappedGymEnv(gym.make("CarlaRlEnv-v0", params=self.params), self.args)

        self.writer = SummaryWriter(log_dir=f"./results/{file_name}")
        self.args.summary_writer = self.writer

        self.args.action_shape = self.env.action_space.shape[0]
        self.args.action_scale = self.env.action_scale
        self.args.action_bias = self.env.action_bias

        # Set seeds
        self.env.action_space.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)

        print(self.args)

        self.policy = SAC(self.args, device)
        self.buffer = PER_Buffer(self.args, device=device)

        self.BETA = self.args.beta_init
        self.BUFFER_ALPHA = 0.6

        self.beta_scheduler = LinearSchedule(self.args.beta_gain_steps, self.args.beta_init, 1.0)
        self.buffer_alpha_scheduler = adaptiveSchedule(self.args.alpha_max, self.args.alpha_min)

        self.actor_lr_scheduler = LinearSchedule(self.args.lr_decay_steps, self.args.lr_init, self.args.lr_end)
        self.critic_lr_scheduler = LinearSchedule(self.args.lr_decay_steps, self.args.lr_init, self.args.lr_end)

        # Load the model if needed
        if self.args.Loadmodel:
            self.policy.load(f"./models/{file_name}")

    def train(self):
        FLAG = True  # 삭제해야하는 코드

        for t in range(int(self.args.max_timesteps)):

            state = self.env.reset()

            done = False
            episode_reward = 0
            episode_cost = 0

            while True:
                if not self.args.Loadmodel and t < self.args.start_timesteps:
                    action = self.policy.select_action(state, random_sample=True)
                else:
                    action = self.policy.select_action(state, random_sample=False)

                # Perform action
                next_state, reward, done, info = self.env.step(action)
                self.env.display()
                cost = info['cost']  # collision & invasion cost

                experience = Experience(state, action, reward, next_state, done)
                self.buffer.add(experience)

                if done:
                    print(f"done after {t+1} steps done is {done}")
                    break

                # Train policy
                if self.policy.has_enough_experience(self.buffer) and t > self.args.start_timesteps:
                    self.policy.train(self.BETA, self.BUFFER_ALPHA, self.buffer)
                    self.BETA = self.beta_scheduler.value(t)
                    td_mean = np.mean(self.buffer.buffer[:len(self.buffer)]["priority"])
                    td_std = np.std(self.buffer.buffer[:len(self.buffer)]["priority"])
                    self.BUFFER_ALPHA = self.buffer_alpha_scheduler.value(td_mean, td_std)

                    if FLAG:
                        print(f"\ntimestep {t} train start beta {self.BETA} td_mean {td_mean} td std {td_std} buffer alpha {self.BUFFER_ALPHA}\n")

                    # Actor lr scheduler
                    for p in self.policy.actor_optimizer.param_groups:
                        p['lr'] = self.actor_lr_scheduler.value(t)

                    # Critic lr scheduler
                    for p in self.policy.critic_optimizer.param_groups:
                        p['lr'] = self.critic_lr_scheduler.value(t)

                    for p in self.policy.critic_optimizer2.param_groups:
                        p['lr'] = self.critic_lr_scheduler.value(t)

                FLAG = False  # 삭제해야하는 코드
                state = next_state
                episode_reward += reward
                episode_cost += cost

                # Evaluate episode
                if (t + 1) % self.args.eval_freq == 0:
                    print("\nEvaluate score\n")
                    avg_reward, avg_cost = eval_policy(self.policy, self.env)
                    if self.args.write:
                        self.args.summary_writer.add_scalar('eval reward', avg_reward, global_step=t)
                        self.args.summary_writer.add_scalar('episode_reward', episode_reward, global_step=t)
                        self.args.summary_writer.add_scalar('eval_cost', avg_cost, global_step=t)
                        self.args.summary_writer.add_scalar('episode_cost', episode_cost, global_step=t)
                        self.args.summary_writer.add_scalar('BUFFER_alpha', self.BUFFER_ALPHA, global_step=t)
                        self.args.summary_writer.add_scalar('beta', self.BETA, global_step=t)

                    self.policy.save(f"./models/{self.file_name}")
                    print('writer add scalar and save model   ', 'steps: {}k'.format(int(t / 1000)), 'AVG reward:', int(avg_reward), 'AVG cost:', int(avg_cost))
                    t = t + 1

            # 10 episode 후 서버로 weight 값 보내기
            if (t + 1) % self.args.update_freq == 0:
                weights = {}
                for name, param in self.policy.actor.named_parameters():
                    weights[name] = param.detach().cpu().numpy()

                parameter_server = ray.get_actor("ParameterServer")
                parameter_server.add_weights.remote(weights)
                print(f"{self.id}'s Weights sent to the central server.")

            print(f"\n--------{self.id}--- timestep : {t} reward : {episode_reward}  cost : {episode_cost}--------\n")

        


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--render', type=bool, default=False,
                        help='Render or Not , render human mode for test, rendoer rgb array for train')

    parser.add_argument('--action_repeat', default=4)
    parser.add_argument('--image_size', default=64)
    parser.add_argument('--seed', default=0)
    parser.add_argument('--buffer_size', default=int(1e6))
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--time_out', default=60.0)

    parser.add_argument('--tau', default=5e-3)
    parser.add_argument('--no_render', default=True)

    parser.add_argument("--alpha_min", default=0.3, type=float)  # PER buffer alpha
    parser.add_argument("--alpha_max", default=0.9, type=float)  # PER buffer alpha

    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')

    parser.add_argument("--start_timesteps", default=2000, type=int)  # Time steps initial random policy is used 2000

    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor

    parser.add_argument("--env", default="CarlaRlEnv-v0")  # register env name
    parser.add_argument('--policy', default="SAC", help='reinforcement algorithm policy ')

    parser.add_argument('--beta_init', type=float, default=0.4, help='beta for PER')
    parser.add_argument('--beta_gain_steps', type=int, default=int(1e6), help='steps of beta from beta_init to 1.0')
    parser.add_argument('--lr_init', type=float, default=2e-4, help='Initial Learning rate')
    parser.add_argument('--lr_end', type=float, default=6e-5, help='Final Learning rate')
    parser.add_argument('--lr_decay_steps', type=float, default=int(1e6), help='Learning rate decay steps')
    parser.add_argument('--write', type=bool, default=True, help='summary T/F')

    parser.add_argument("--update_freq", default=10, type=int)  # How often we update central model
    parser.add_argument("--eval_freq", default=1e3, type=int)  # How often (time steps) we evaluate 1e3

    parser.add_argument('--Loadmodel', type=bool, default=False,help='Load pretrained model or Not')  # 훈련 마치고 나서는 True로 설정 하기

    parser.add_argument('--port', type=int, default=3000, help='Port num')
    parser.add_argument('--traffic_port', type=int, default=8000, help='Traffic port num')

    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists(f"./results/{file_name}"):
        os.makedirs(f"./results/{file_name}")

    if not os.path.exists(f"./models/{file_name}"):
        os.makedirs(f"./models/{file_name}")

    # carla env parameter
    params = {
        'carla_port': args.port,
        'traffic_port': args.traffic_port,
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
        'sensors_to_amount': ['front_rgb', 'lidar'],
    }

    num_workers = 2

    ray.shutdown()
    ray.init(address='auto', namespace="parameter_server_namespace")
    workers = []
    for worker_id in range(num_workers):
        port = args.port + worker_id * 100
        print(port)
        traffic_port = args.traffic_port + worker_id * 100
        worker = Worker.remote(worker_id, params, args, file_name, port, traffic_port)
        workers.append(worker)

    # 각 워커에서 학습 시작
    train_ids = [worker.train.remote() for worker in workers]
    ray.get(train_ids)


if __name__ == '__main__':
    main()


