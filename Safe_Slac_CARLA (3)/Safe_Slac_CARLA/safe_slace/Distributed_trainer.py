import os
from collections import deque
from time import sleep, time
import torch
from ray.tune.examples.pbt_dcgan_mnist.common import image_size

from safe_slace.ReplayBuffer import CostReplayBuffer
from gym.spaces.box import Box
from safe_slace.algo import LatentPolicySafetyCriticSlac

from WrappedGymEnv import WrappedGymEnv
import gym
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ray import train
import ray
from ray.train import get_context
from gym.envs.registration import register
from carla_rl_env.carla_env import CarlaRlEnv
import gc
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy



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


def create_carla_env(params):
    return CarlaRlEnv.remote(params)


def train_loop(config):
    # initialize config setting
    log_dir = config["log_dir"]

    # algo_config = config["model_config"]
    model_dir = config["model_dir"]
    training_config = config["training_config"]

    # env = config["env"]
    # env_test = config["env_test"]

    num_sequences = training_config.get("num_sequences")
    action_repeat = training_config.get("action_repeat")
    eval_interval = training_config.get("eval_interval")
    gamma_c = training_config.get("gamma_c")
    num_eval_episodes = training_config.get("num_eval_episodes")
    device = training_config.get("device")
    seed = training_config.get("seed")
    buffer_size = training_config.get("buffer_size")
    feature_dim = training_config.get("feature_dim")
    z2_dim = training_config.get("z2_dim")
    hidden_units = training_config.get("hidden_units")
    batch_size_latent = training_config.get("batch_size_latent")
    batch_size_sac = training_config.get("batch_size_sac")
    lr_sac = training_config.get("lr_sac")
    lr_latent = training_config.get("lr_latent")
    start_alpha = training_config.get("start_alpha")
    start_lagrange = training_config.get("start_lagrange")
    grad_clip_norm = training_config.get("grad_clip_norm")
    tau = training_config.get("tau")
    image_noise = training_config.get("image_noise")
    image_size = training_config.get("image_size")

    # initialize wirter of each worker logging

    worker_rank = get_context().get_world_rank()  # bound : 0 ~ num_worker-1


    writer = SummaryWriter(logdir=os.path.join(log_dir, f'_{worker_rank}'))

    pg_buffer = placement_group([{"CPU": 2, "GPU": 0.125}],strategy="PACK")
    ray.get(pg_buffer.ready())
    buffer = CostReplayBuffer.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg_buffer)).remote(buffer_size, num_sequences, device, image_size=image_size)

    initial_collection_steps = training_config.get("initial_collection_steps")
    initial_learning_steps = training_config.get("initial_learning_steps")
    collect_with_policy = training_config.get("collect_with_policy")
    train_steps_per_iter = training_config.get("train_steps_per_iter")
    num_steps = training_config.get("num_steps")

    # env params
    params = {
        'carla_port': 2000,
        'map_name': 'Town10HD',
        'window_resolution': [720, 720],
        'grid_size': [3, 3],
        'sync': True,
        'no_render': True,
        'display_sensor': False,
        'ego_filter': 'vehicle.tesla.model3',
        'num_vehicles': 30,
        'num_pedestrians': 2,
        'enable_route_planner': True,
        'sensors_to_amount': ['left_rgb', 'front_rgb', 'right_rgb', 'top_rgb', 'lidar', 'radar'],
        'image_size': image_size,
        "worker_rank":worker_rank
    }

    register(
        id='CarlaRlEnv-v0',
        entry_point='carla_rl_env.carla_env:CarlaRlEnv'  # create_carla_env#'carla_rl_env.carla_env:CarlaRlEnv',
    )

    pg_algo = placement_group([{"CPU": 3, "GPU": 0.25}],strategy="PACK")
    ray.get(pg_algo.ready())
    algo = LatentPolicySafetyCriticSlac.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg_algo)).remote(
        num_sequences=num_sequences,
        gamma_c=gamma_c,
        state_shape=Box(0, 255, (18, 64, 64), np.uint8).shape,
        ometer_shape=Box(-np.inf, np.inf, shape=(40, 2), dtype=np.float32).shape,
        tgt_state_shape=Box(0, 255, (3, 64, 64), np.uint8).shape,
        action_shape=Box(-1.0, 1.0, shape=(2,)).shape,
        action_repeat=action_repeat,
        device=device,
        seed=seed,
        buffer_size=buffer_size,
        feature_dim=feature_dim,
        z2_dim=z2_dim,
        hidden_units=hidden_units,
        batch_size_latent=batch_size_latent,
        batch_size_sac=batch_size_sac,
        lr_sac=lr_sac,
        lr_latent=lr_latent,
        start_alpha=start_alpha,
        start_lagrange=start_lagrange,
        grad_clip_norm=grad_clip_norm,
        tau=tau,
        image_noise=image_noise)

    #ray.get(algo.load_model.remote("tmp"))

    register(
        id='CarlaRlEnv-v0',
        entry_point='carla_rl_env.carla_env:CarlaRlEnv'  # create_carla_env#'carla_rl_env.carla_env:CarlaRlEnv',
    )
    # env = CarlaRlEnv.remote(params)  # WrappedGymEnv.remote(gym.make('CarlaRlEnv-v0', params=params), action_repeat=4, image_size=64)
    # Episode timestep
    t = 0

    # carla_env = CarlaRlEnv.remote(params)

    env = CarlaRlEnv.remote(params)  # WrappedGymEnv.remote(gym.make('CarlaRlEnv-v0', params=params), action_repeat=4, image_size=64)
    # env_test = env

    env_shape_ref = env.get_env_shape.remote()  # algo.get_env_shape.remote()
    env_shape = ray.get(env_shape_ref)

    # env_shape = env.get_env_shape()

    ob = SlacObservation(env_shape["env_observation_space_shape"], env_shape["env_ometer_space_shape"],
                         env_shape["env_tgt_state_space_shape"], env_shape["env_action_space_shape"], num_sequences)

    # ob_test = SlacObservation(env_shape["env_observation_space_shape"], env_shape["env_ometer_space_shape"], env_shape["env_tgt_state_space_shape"],env_shape["env_action_space_shape"], num_sequences)

    # ob = SlacObservation(env.observation_space.shape,env.ometer_space.shape,env.tgt_state_space.shape,env.action_space.shape, num_sequences)
    #
    # ob_test = SlacObservation(env.observation_space.shape,env.ometer_space.shape,env.tgt_state_space.shape,env.action_space.shape, num_sequences)

    # initialize env reset
    env_reset_ref = env.reset.remote()  # algo.get_env_reset.remote()
    env_reset = ray.get(env_reset_ref)

    state, ometer, tgt_state = env_reset  # env.reset()
    ob.reset_episode(state, ometer, tgt_state)

    buffer.reset_episode.remote(state, ometer, tgt_state)

    # Collect trajectories using random policy.
    bar = tqdm(range(1, initial_collection_steps + 1))

    for step in bar:
        bar.set_description("collect data.")
        t += 1
        pid_sample = env.pid_sample.remote()
        action = ray.get(pid_sample)
        #action = ray.put(action)
        env_step = env.step.remote(action)
        state, ometer, tgt_state, reward, done, info = ray.get(env_step)
        cost = info["cost"]
        algo.set_cost_reward_return.remote(cost, reward)
        #ray.get(algo.set_cost_reward_return.remote(cost, reward))
        mask = False if t >= 1000 else done

        ob.append(state, ometer, tgt_state, action)

        buffer.append.remote(action, reward, mask, state, ometer, tgt_state, done, cost)

        if done:
            t = 0
            env_reset = env.reset.remote()
            state, ometer, tgt_state = ray.get(env_reset)  # self.env.reset()

            ob.reset_episode(state, ometer, tgt_state)
            buffer.reset_episode.remote(state, ometer, tgt_state)

            #ray.get(algo.set_z1_z2_none.remote())  # set self.z1 z2 none
            algo.set_z1_z2_none.remote()

     
    #print("done cd action")

    # Update latent variable model first so that SLAC can learn well using (learned) latent dynamics.

    bar = tqdm(range(initial_learning_steps * 2))
    latent_batch_refs = []

    for _ in bar:
        bar.set_description("pre-update latent.")
        latent_batch_ref = buffer.sample_latent.remote(batch_size_latent)
        latent_batch_refs.append(latent_batch_ref)

    unfinished = latent_batch_refs
     

    latent_gradient_ref = []

    #batch = None
    # for unfin in range(len(unfinished)):
    #     if len(latent_gradient_ref) > 100:
    #         done_ref, latent_gradient_ref = ray.wait(latent_gradient_ref, num_returns=1)
    #         batch = ray.get(done_ref)
    #         batch = ray.put(batch)
    #
    #         ray.get(algo.update_latent.remote(batch))
    #         del batch
    #         ray.internal.free([done_ref])
    #     latent_gradient_ref.append(unfin)

    #ray.get(latent_gradient_ref)

    #print("done latent batch")


    while unfinished:
        done_ref, unfinished = ray.wait(unfinished, num_returns=1)
        #batch = ray.get(done_ref[0])

        #batch = ray.put(batch)
        #latent_gradient_ref.append(algo.update_latent.remote(batch))
        algo.update_latent.remote(done_ref)
        #ray.internal.free([done_ref[0]])
        #ray.get(ref)
        #del batch
        #del ref


    #
    #
    #
    print("done latent batch")
    # unfinished_latent_grad = latent_gradient_ref
    #
    # while unfinished_latent_grad:
    #     done_ref ,unfinished_ = ray.wait(unfinished_latent_grad,num_returns=1)
    #     ray.get(done_ref[0])
    #     ray.internal.free([done_ref[0]])

     
    print("done latent gradient")

    bar = tqdm(range(initial_learning_steps))
    sac_batch_refs = []
    sac_gradient_ref = []
    for _ in bar:
        bar.set_description("pre-update sac.")
        sac_batch_ref = buffer.sample_sac.remote(batch_size_sac)
        sac_batch_refs.append(sac_batch_ref)

    unfinished = sac_batch_refs
    while unfinished:
        done_ref, unfinished = ray.wait(unfinished, num_returns=1)

        algo.update_sac.remote(done_ref)


    save_dir = os.path.join(model_dir)
    save_ref = algo.save_model.remote()
    state_dicts =ray.get(save_ref)
    torch.save(state_dicts["encoder"], os.path.join(save_dir, "encoder.pth"))
    torch.save(state_dicts["decoder"], os.path.join(save_dir, "decoder.pth"))
    torch.save(state_dicts["latent"], os.path.join(save_dir, "latent.pth"))
    torch.save(state_dicts["actor"], os.path.join(save_dir, "actor.pth"))
    torch.save(state_dicts["critic"], os.path.join(save_dir, "critic.pth"))
    torch.save(state_dicts["critic_target"], os.path.join(save_dir, "critic_target.pth"))
    torch.save(state_dicts["safety_critic"], os.path.join(save_dir, "safety_critic.pth"))
    torch.save(state_dicts["safety_critic_target"], os.path.join(save_dir, "safety_critic_target.pth"))
    print("done model save")
    # 

    # Iterate collection, update and evaluation.

    bar = tqdm(range(initial_collection_steps + 1, num_steps // action_repeat + 1))

    episode_reward_list=[]
    episode_cost_list=[]
    episode_reward=0
    episode_cost =0
    for step in bar:
        bar.set_description("training part")
        t += 1

        explore_action = algo.explore.remote(ob)
        action = ray.get(explore_action)
        #action = ray.put(action)
        env_step = env.step.remote(action)
        state, ometer, tgt_state, reward, done, info = ray.get(env_step)

        cost = info["cost"]
        #ray.get(algo.set_cost_reward_return.remote(cost, reward))
        algo.set_cost_reward_return.remote(cost, reward)
        episode_reward+=reward
        episode_cost+=cost

        mask = False if t >= 1000 else done

        ob.append(state, ometer, tgt_state, action)

        buffer.append.remote(action, reward, mask, state, ometer, tgt_state, done, cost)

        if done:
            t = 0
            env_reset = env.reset.remote()
            state, ometer, tgt_state = ray.get(env_reset)  # self.env.reset()

            ob.reset_episode(state, ometer, tgt_state)
            buffer.reset_episode.remote(state, ometer, tgt_state)
            episode_reward_list.append(episode_reward)
            episode_cost_list.append(episode_cost)
            episode_reward=0
            episode_cost=0
            # ray.get(algo.set_z1_z2_none.remote())  # set self.z1 z2 none
            algo.set_z1_z2_none.remote()

        #print("done explore")

        algo.update_lag.remote()

        #print("done update lag")
        print(t)

        # Update the algorithm. t #t+1
        if step % train_steps_per_iter == 0:
            #print("update lat and sac in training loop")

            latent_batch_refs = []
            sac_batch_refs = []
            for _ in range(train_steps_per_iter):
                latent_batch_ref = buffer.sample_latent.remote(batch_size_latent)
                sac_batch_ref = buffer.sample_sac.remote(batch_size_sac)

                latent_batch_refs.append(latent_batch_ref)
                sac_batch_refs.append(sac_batch_ref)

            unfinished = latent_batch_refs
            latent_gradient_ref = []
            while unfinished:
                done_ref, unfinished = ray.wait(unfinished, num_returns=1)

                algo.update_latent.remote(done_ref)

            #print("done latent update in training loop")
            unfinished = sac_batch_refs
            while unfinished:
                done_ref, unfinished = ray.wait(unfinished, num_returns=1)

                algo.update_sac.remote(done_ref)

            #print("done update sac in training loop")
        # Evaluate regularly.
        step_env = step * action_repeat  #eval_interval
        if step_env % eval_interval == 0:
            print("start evaluate ")
            save_dir = os.path.join(model_dir)
            save_ref = algo.save_model.remote()
            state_dicts = ray.get(save_ref)
            torch.save(state_dicts["encoder"], os.path.join(save_dir, "encoder.pth"))
            torch.save(state_dicts["decoder"], os.path.join(save_dir, "decoder.pth"))
            torch.save(state_dicts["latent"], os.path.join(save_dir, "latent.pth"))
            torch.save(state_dicts["actor"], os.path.join(save_dir, "actor.pth"))
            torch.save(state_dicts["critic"], os.path.join(save_dir, "critic.pth"))
            torch.save(state_dicts["critic_target"], os.path.join(save_dir, "critic_target.pth"))
            torch.save(state_dicts["safety_critic"], os.path.join(save_dir, "safety_critic.pth"))
            torch.save(state_dicts["safety_critic_target"], os.path.join(save_dir, "safety_critic_target.pth"))


            loss_values_ref = algo.get_loss_values.remote()
            stats_ref = algo.get_stats.remote()


            loss_values = ray.get(loss_values_ref)
            stats = ray.get(stats_ref)


            metrics = {"cost/train":np.mean(episode_cost_list),
                       "return/train":np.mean(episode_reward_list)}
            train.report(metrics)
            episode_reward_list=[]
            episode_cost_list=[]


            train.report(loss_values)
            train.report(stats)


        if step_env % 1000 == 0:
            ray.get(algo.step_schedulers.remote())

        epoch_len_ref = algo.get_epoch_len.remote()
        epoch_len = ray.get(epoch_len_ref)

        if step_env % epoch_len == 0:
            pass




    # Wait for logging to be finished.
    sleep(10)

















