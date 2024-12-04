import argparse

import torch

from safe_slace.algo import LatentPolicySafetyCriticSlac

from WrappedGymEnv import WrappedGymEnv
import gym

from configuration import get_default_config
import numpy as np
from collections import deque

class SlacObservation:
    """
    Observation for SLAC.
    """

    def __init__(self, state_shape, ometer_shape, tgt_state_shape, action_shape, num_sequences):
        self.state_shape = state_shape
        self.ometer_shape = ometer_shape
        self.tgt_state_shape = tgt_state_shape
        self.action_shape = action_shape
        self.num_sequences = num_sequences

    def reset_episode(self, state, ometer, tgt_state):
        self._state = deque(maxlen=self.num_sequences)
        self._ometer = deque(maxlen=self.num_sequences)
        self._tgt_state = deque(maxlen=self.num_sequences)
        self._action = deque(maxlen=self.num_sequences - 1)
        for _ in range(self.num_sequences - 1):
            self._state.append(np.zeros(self.state_shape, dtype=np.uint8))
            self._ometer.append(np.zeros(self.ometer_shape, dtype=np.float32))
            self._tgt_state.append(np.zeros(self.tgt_state_shape, dtype=np.uint8))
            self._action.append(np.zeros(self.action_shape, dtype=np.float32))
        self._state.append(state)
        self._ometer.append(ometer)
        self._tgt_state.append(tgt_state)

    def append(self, state, ometer, tgt_state, action):
        self._state.append(state)
        self._ometer.append(ometer)
        self._tgt_state.append(tgt_state)
        self._action.append(action)

    @property
    def state(self):
        return np.array(self._state)[None, ...]

    @property
    def last_state(self):
        return np.array(self._state[-1])[None, ...]

    @property
    def ometer(self):
        return np.array(self._ometer)[None, ...]

    @property
    def last_ometer(self):
        return np.array(self._ometer[-1])[None, ...]

    @property
    def tgt_state(self):
        return np.array(self._tgt_state)[None, ...]

    @property
    def last_tgt_state(self):
        return np.array(self._tgt_state[-1])[None, ...]

    @property
    def action(self):
        return np.array(self._action).reshape(1, -1)

    @property
    def last_action(self):
        return np.array(self._action[-1])

def inference(args):
    config = get_default_config()
    config["domain_name"] = args.domain_name
    config["task_name"] = args.task_name
    config["seed"] = args.seed
    config["num_steps"] = args.num_steps
    # env params
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
    from gym.envs.registration import register

    register(
        id='CarlaRlEnv-v0',
        entry_point='carla_rl_env.carla_env:CarlaRlEnv',
    )

    env = WrappedGymEnv(gym.make(args.domain_name, params=params),
                         action_repeat=args.action_repeat,image_size=64)

    algo = LatentPolicySafetyCriticSlac(
        num_sequences=config["num_sequences"],
        gamma_c=config["gamma_c"],
        state_shape=env.observation_space.shape,
        ometer_shape=env.ometer_space.shape,
        tgt_state_shape=env.tgt_state_space.shape,
        action_shape=env.action_space.shape,
        action_repeat=config["action_repeat"],
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=config["seed"],
        buffer_size=config["buffer_size"],
        feature_dim=config["feature_dim"],
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
    algo.load_model(args.load_dir)
    num_eval_episodes = args.num_eval_episodes

    ob = SlacObservation(env.observation_space.shape, env.ometer_space.shape, env.tgt_state_space.shape,
                              env.action_space.shape, 10)

    reward_returns = []
    cost_returns = []
    for i in range(num_eval_episodes):
        algo.z1 = None
        algo.z2 = None

        state,ometer,tgt_state = env.reset()
        ob.reset_episode(state,ometer, tgt_state)

        episode_return = 0.0
        cost_return = 0.0
        done = False

        while not done:
            action =  algo.explore(ob) # env.pid_sample()
            state, ometer, tgt_state, reward, done, info = env.step(action)
            env.display()
            cost = info["cost"]
            ob.append(state, ometer, tgt_state, action)
            episode_return += reward
            cost_return += cost

        reward_returns.append(episode_return)
        cost_returns.append(cost_return)

    mean_reward_return = np.mean(reward_returns)
    mean_cost_return = np.mean(cost_returns)



    algo.z1=None
    algo.z2 = None
    print(f"Steps:    " f"Return: {mean_reward_return:<5.1f} " f"CostRet: {mean_cost_return:<5.1f}   ")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=2 * 10 ** 6)
    parser.add_argument("--domain_name", type=str, default="CarlaRlEnv-v0")
    parser.add_argument("--task_name", type=str, default="run")
    parser.add_argument("--action_repeat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=str,default="cuda")

    parser.add_argument("--num_eval_episodes", type=int,default=5)
    parser.add_argument("--load_dir", type=str, default="logs/CarlaRlEnv-v0-run/slac-seed0-20241201-1119/models/step20000")

    args = parser.parse_args()
    inference(args)