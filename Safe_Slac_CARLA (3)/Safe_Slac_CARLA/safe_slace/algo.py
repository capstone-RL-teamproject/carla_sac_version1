import os

from ray.util import placement_group

from WrappedGymEnv import WrappedGymEnv
import numpy as np
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
import gym

from safe_slace.network import GaussianPolicy, TwinnedQNetwork, SingleQNetwork,LatentGaussianPolicy
from safe_slace.latent import CostLatentModel

from utils import create_feature_actions, grad_false, soft_update,sample_reproduction
from collections import defaultdict
import torch.nn.functional
import ray
# from ray.util.placement_group import placement_group
#
# pg = placement_group([{"CPU": 1, "GPU": 0.125}],strategy="PACK")
# ray.get(pg.ready())  # placement group이 준비될 때까지 대기

@ray.remote(num_cpus=3,num_gpus=0.25)#(num_cpus=1, num_gpus=0.125,placement_group=pg)
class LatentPolicySafetyCriticSlac:
    """
    Latent state-based safe SLAC algorithm.
    """

    def __init__(
            self,
            state_shape,
            ometer_shape,
            tgt_state_shape,
            action_shape,
            action_repeat,
            device,
            seed,
            gamma=0.99,
            gamma_c=0.995,
            batch_size_sac=256,
            batch_size_latent=32,
            buffer_size=10 ** 5,
            num_sequences=8,
            lr_sac=3e-4,
            lr_latent=1e-4,
            feature_dim=256,
            z1_dim=32,
            z2_dim=256,
            hidden_units=(256, 256),
            tau=5e-3,
            start_alpha=3.3e-4,
            start_lagrange=2.5e-2,
            grad_clip_norm=10.0,
            image_noise=0.1
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # from gym.envs.registration import register
        #
        # register(
        #     id='CarlaRlEnv-v0',
        #     entry_point='carla_rl_env.carla_env:CarlaRlEnv',
        # )
        #
        # self.env = WrappedGymEnv(gym.make("CarlaRlEnv-v0", params=params), action_repeat=4, image_size=64)
        # self.env_test = self.env

        budget = 25
        self.budget_undiscounted = budget
        self.steps = 1000 / action_repeat
        self.budget = budget * (1 - gamma_c ** (1000 / action_repeat)) / (1 - gamma_c) / (1000 / action_repeat)


        self.grad_clip_norm = grad_clip_norm
        # Networks.

        self.actor = LatentGaussianPolicy(action_shape, z1_dim, z2_dim, hidden_units)
        self.critic = TwinnedQNetwork(action_shape, z1_dim, z2_dim, hidden_units)
        self.critic_target = TwinnedQNetwork(action_shape, z1_dim, z2_dim, hidden_units)
        self.safety_critic = SingleQNetwork(action_shape, z1_dim, z2_dim, hidden_units, init_output=self.budget)
        self.safety_critic_target = SingleQNetwork(action_shape, z1_dim, z2_dim, hidden_units, init_output=self.budget)
        self.latent = CostLatentModel(state_shape, ometer_shape, tgt_state_shape, action_shape, feature_dim, z1_dim,
                                      z2_dim, hidden_units, image_noise=image_noise)
        soft_update(self.critic_target, self.critic, 1.0)
        soft_update(self.safety_critic_target, self.safety_critic, 1.0)

        parts = [(self.actor, None, "actor"),
                 (self.critic, None, "critic"),
                 (self.critic_target, None, "critic_target"),
                 (self.safety_critic, None, "safety_critic"),
                 (self.safety_critic_target, None, "safety_critic_target"),
                 (self.latent, None, "latent")]
        for model, optimizer, name in parts:
            model.to(device)
            if "target" not in name:
                model.train()

        grad_false(self.critic_target)
        grad_false(self.safety_critic_target)

        # Target entropy is -|A|.
        self.target_entropy = -np.prod(action_shape) * 1.0
        # We optimize log(alpha) because alpha is always bigger than 0.
        self.log_alpha = torch.tensor([np.log(start_alpha)], requires_grad=True, device=device, dtype=torch.float32)
        with torch.no_grad():
            self.alpha = self.log_alpha.exp()

        self.raw_lag = torch.tensor([np.log(np.exp(start_lagrange) - 1)], requires_grad=True, device=device,
                                    dtype=torch.float32)
        with torch.no_grad():
            self.lagrange = torch.nn.functional.softplus(self.raw_lag)
        # Optimizers.
        self.optim_actor = Adam(self.actor.parameters(), lr=lr_sac)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_sac)
        self.optim_safety_critic = Adam(self.safety_critic.parameters(), lr=lr_sac)
        self.optim_alpha = Adam([self.log_alpha], lr=lr_sac)
        self.optim_lagrange = SGD([self.raw_lag], lr=2e-4)
        self.optim_latent = Adam(self.latent.parameters(), lr=lr_latent)

        self.sched_actor = MultiStepLR(self.optim_actor, milestones=[400], gamma=0.5)
        self.sched_critic = MultiStepLR(self.optim_critic, milestones=[400], gamma=0.5)
        self.sched_safety_critic = MultiStepLR(self.optim_safety_critic, milestones=[400], gamma=0.5)
        self.sched_alpha = MultiStepLR(self.optim_alpha, milestones=[400], gamma=0.5)
        self.sched_lagrange = MultiStepLR(self.optim_lagrange, milestones=[400], gamma=0.5)
        self.sched_latent = MultiStepLR(self.optim_latent, milestones=[400], gamma=0.5)

        self.scheds = [self.sched_actor, self.sched_critic, self.sched_safety_critic, self.sched_alpha,
                       self.sched_lagrange, self.sched_latent]

        self.learning_steps_sac = 0
        self.learning_steps_latent = 0
        self.state_shape = state_shape
        self.ometer_shape = ometer_shape
        self.tgt_state_shape = tgt_state_shape
        self.action_shape = action_shape
        self.action_repeat = action_repeat
        self.device = device
        self.gamma = gamma
        self.gamma_c = gamma_c
        self.batch_size_sac = batch_size_sac
        self.batch_size_latent = batch_size_latent
        self.num_sequences = num_sequences
        self.tau = tau

        self.epoch_len = 30_000 // self.action_repeat
        self.epoch_costreturns = []
        self.epoch_rewardreturns = []
        self.episode_costreturn = 0
        self.episode_rewardreturn = 0

        self.loss_averages = defaultdict(lambda: 0)

        self.loss_image = torch.tensor(0.0)
        self.loss_actor = torch.tensor(0.0)
        self.loss_kld = torch.tensor(0.0)
        self.loss_reward = torch.tensor(0.0)
        self.loss_critic = torch.tensor(0.0)
        self.loss_safety_critic = torch.tensor(0.0)
        self.loss_cost = torch.tensor(0.0)
        self.loss_alpha = torch.tensor(0.0)
        self.loss_lag = torch.tensor(0.0)
        self.entropy = torch.tensor(0.0)

        # JIT compile to speed up.
        # fake_feature = torch.empty(1, num_sequences + 1, feature_dim, device=device)
        # fake_action = torch.empty(1, num_sequences, action_shape[0], device=device)
        # self.create_feature_actions = torch.jit.trace(create_feature_actions, (fake_feature, fake_action))
        self.z1 = None
        self.z2 = None
        self.create_feature_actions = create_feature_actions


    def set_z1_z2_none(self):
        self.z1=None
        self.z2=None

    def set_cost_reward_return(self,cost,reward):
        self.lastcost = cost
        self.episode_costreturn += cost
        self.episode_rewardreturn += reward

    def get_env_shape(self):
        return {
            "env_observation_space_shape":self.env.observation_space.shape,
            "env_ometer_space_shape":self.env.ometer_space.shape,
            "env_tgt_state_space_shape":self.env.tgt_state_space.shape,
            "env_action_space_shape":self.env.action_space.shape
        }
    def get_epoch_costreturns(self):
        return self.epoch_costreturns

    def get_epoch_rewardreturns(self):
        return self.epoch_rewardreturns

    def reset_epoch_returns(self):
        self.epoch_costreturns=[]
        self.epoch_rewardreturns=[]

    def get_loss_values(self):
        # Return a dictionary of loss values
        return {
            'image': self.loss_image.item(),
            'actor': self.loss_actor.item(),
            'kld': self.loss_kld.item(),
            'reward': self.loss_reward.item(),
            'critic': self.loss_critic.item(),
            'safety_critic': self.loss_safety_critic.item(),
            'cost': self.loss_cost.item(),
            'alpha': self.loss_alpha.item(),
            'lag': self.loss_lag.item()
        }

    def get_stats(self):
        # Return a dictionary of statistics
        return {
            'alpha': self.alpha.item(),
            'entropy': self.entropy.item(),
            'lag': self.lagrange.item()
        }

    def step_schedulers(self):
        for sched in self.scheds:
            sched.step()

    def get_epoch_len(self):
        return self.epoch_len



    def preprocess(self, ob):
        state = torch.tensor(ob.last_state, dtype=torch.uint8, device=self.device).float().div_(255.0)
        ometer = torch.tensor(ob.last_ometer, dtype=torch.float32, device=self.device).float().div_(150.0)
        with torch.no_grad():
            feature = self.latent.encoder(state.unsqueeze(0), ometer.unsqueeze(0))
        action = torch.tensor(ob.last_action, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

        if self.z2 is None:
            z1_mean, z1_std = self.latent.z1_posterior_init(feature)
            self.z1 = z1_mean + torch.randn_like(z1_std) * z1_std
            z2_mean, z2_std = self.latent.z2_posterior_init(self.z1)
            self.z2 = z2_mean + torch.randn_like(z2_std) * z2_std
        else:
            z1_mean, z1_std = self.latent.z1_posterior(torch.cat([feature, self.z2, action], dim=-1))
            self.z1 = z1_mean + torch.randn_like(z1_std) * z1_std
            # q(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean, z2_std = self.latent.z2_posterior(torch.cat([self.z1, self.z2, action], dim=-1))
            self.z2 = z2_mean + torch.randn_like(z2_std) * z2_std

        return torch.cat([self.z1, self.z2], dim=-1)

    def explore(self, ob):
        z = self.preprocess(ob)
        with torch.no_grad():
            action = self.actor.sample(z)[0][0]
        return action.cpu().numpy()[0]

    def exploit(self, ob):
        z = self.preprocess(ob)
        with torch.no_grad():
            action = self.actor(z)
        return action.cpu().numpy()[0][0]

    def update_actor(self, z, feature_action):
        action, log_pi = self.actor.sample(z)
        q1, q2 = self.critic(z, action)
        c1 = self.safety_critic(z, action)

        with torch.no_grad():
            budget_diff = (self.budget - c1)
            budget_remainder = budget_diff.mean()

        self.loss_actor = -torch.mean(torch.min(q1, q2) - self.alpha * log_pi - (self.lagrange.detach()) * c1, dim=0)

        self.optim_actor.zero_grad()
        self.loss_actor.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm * 2)
        self.optim_actor.step()

        with torch.no_grad():
            self.entropy = -log_pi.detach().mean()
        self.loss_alpha = -self.log_alpha * (self.target_entropy - self.entropy)

        self.optim_alpha.zero_grad()
        self.loss_alpha.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.log_alpha, self.grad_clip_norm)
        self.optim_alpha.step()
        with torch.no_grad():
            self.alpha = self.log_alpha.exp()

    def update_lag(self):
        try:
            last_cost = self.lastcost
        except:
            print("update lag ERROR Exception\n")
            return
        self.loss_lag = (
                    torch.nn.functional.softplus(self.raw_lag) / torch.nn.functional.softplus(self.raw_lag).detach() * (
                        self.budget_undiscounted / self.steps - last_cost)).mean()

        self.optim_lagrange.zero_grad()
        self.loss_lag.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.raw_lag, self.grad_clip_norm * 50)

        self.optim_lagrange.step()
        with torch.no_grad():
            self.lagrange = torch.nn.functional.softplus(self.raw_lag)

    def update_sac(self, batch):
        batch = ray.get(batch[0])
        if batch is None:
            raise ValueError("Batch is None ERROR")


        self.learning_steps_sac += 1
        state_, ometer_, tgt_state_, action_, reward, done, cost = batch
        z, next_z, action, feature_action, next_feature_action = self.prepare_batch(state_, ometer_, action_)

        self.update_critic(z, next_z, action, next_feature_action, reward, done)
        self.update_safety_critic(z, next_z, action, next_feature_action, cost, done)
        self.update_actor(z, feature_action)
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.safety_critic_target, self.safety_critic, self.tau)

    def prepare_batch(self,state_,ometer_,action_):

        with torch.no_grad():
            feature_ = self.latent.encoder(state_,ometer_)
            z_ = torch.cat(self.latent.sample_posterior(feature_,action_)[2:4],dim=-1)


        z,next_z = z_[:,-2],z_[:,-1]

        action = action_[:,-1]

        feature_action ,next_feature_action = self.create_feature_actions(feature_,action_)

        return z,next_z,action,feature_action,next_feature_action




    def update_critic(self, z, next_z, action, next_feature_action, reward, done):
        curr_q1, curr_q2 = self.critic(z, action)
        with torch.no_grad():
            next_action, log_pi = self.actor.sample(next_z)
            next_q1, next_q2 = self.critic_target(next_z, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * log_pi
        target_q = reward + (1.0 - done) * self.gamma * next_q
        self.loss_critic = (curr_q1 - target_q).pow_(2).mean() + (curr_q2 - target_q).pow_(2).mean()

        self.optim_critic.zero_grad()
        self.loss_critic.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.optim_critic.step()

    def update_latent(self, batch):
        batch = ray.get(batch[0])
        if batch is None:
            raise ValueError("Batch is None in update latent ERROR")

        self.learning_steps_latent += 1
        state_, ometer_, tgt_state_, action_, reward_, done_, cost_ = batch
        self.loss_kld, self.loss_image, self.loss_reward, self.loss_cost = self.latent.calculate_loss(state_, ometer_, tgt_state_, action_, reward_, done_, cost_)

        self.optim_latent.zero_grad()
        (self.loss_kld + self.loss_image + self.loss_reward + self.loss_cost).backward()
        torch.nn.utils.clip_grad_norm_(self.latent.parameters(), self.grad_clip_norm)
        self.optim_latent.step()

    def update_safety_critic(self, z, next_z, action, next_feature_action, cost, done):
        curr_c1 = self.safety_critic(z, action)
        with torch.no_grad():
            next_action, log_pi = self.actor.sample(next_z)
            next_c = self.safety_critic_target(next_z, next_action)
            target_c = cost + (1.0 - done) * self.gamma_c * next_c
        self.loss_safety_critic = torch.nn.functional.mse_loss(curr_c1, target_c)

        self.optim_safety_critic.zero_grad()
        self.loss_safety_critic.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.safety_critic.parameters(), self.grad_clip_norm)
        self.optim_safety_critic.step()





    def save_model(self):
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        print("save_model in algo")
        # # We don't save target network to reduce workloads.
        # torch.save(self.latent.encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
        # torch.save(self.latent.decoder.state_dict(), os.path.join(save_dir, "decoder.pth"))
        # torch.save(self.latent.state_dict(), os.path.join(save_dir, "latent.pth"))
        # torch.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))
        # torch.save(self.critic.state_dict(), os.path.join(save_dir, "critic.pth"))
        # torch.save(self.critic_target.state_dict(), os.path.join(save_dir, "critic_target.pth"))
        # torch.save(self.safety_critic.state_dict(), os.path.join(save_dir, "safety_critic.pth"))
        # torch.save(self.safety_critic_target.state_dict(), os.path.join(save_dir, "safety_critic_target.pth"))

        state_dict={
            "encoder":self.latent.encoder.state_dict(),
            "decoder":self.latent.decoder.state_dict(),
            "latent":self.latent.state_dict(),
            "actor":self.actor.state_dict(),
            "critic":self.critic.state_dict(),
            "critic_target":self.critic_target.state_dict(),
            "safety_critic":self.safety_critic.state_dict(),
            "safety_critic_target":self.safety_critic_target.state_dict()

        }
        return state_dict

    def load_model(self, load_dir):
        print('E')
        if os.path.exists(os.path.join(load_dir, "encoder.pth")):
            self.latent.encoder.load_state_dict(torch.load(os.path.join(load_dir, "encoder.pth")))
            print('E')
        if os.path.exists(os.path.join(load_dir, "decoder.pth")):
            self.latent.decoder.load_state_dict(torch.load(os.path.join(load_dir, "decoder.pth")))
            print('E')
        if os.path.exists(os.path.join(load_dir, "latent.pth")):
            self.latent.load_state_dict(torch.load(os.path.join(load_dir, "latent.pth")))
        if os.path.exists(os.path.join(load_dir, "actor.pth")):
            self.actor.load_state_dict(torch.load(os.path.join(load_dir, "actor.pth")))
        if os.path.exists(os.path.join(load_dir, "critic.pth")):
            self.critic.load_state_dict(torch.load(os.path.join(load_dir, "critic.pth")))
        if os.path.exists(os.path.join(load_dir, "critic_target.pth")):
            self.critic_target.load_state_dict(torch.load(os.path.join(load_dir, "critic_target.pth")))
        if os.path.exists(os.path.join(load_dir, "safety_critic.pth")):
            self.safety_critic.load_state_dict(torch.load(os.path.join(load_dir, "safety_critic.pth")))
        if os.path.exists(os.path.join(load_dir, "safety_critic_target.pth")):
            self.safety_critic_target.load_state_dict(torch.load(os.path.join(load_dir, "safety_critic_target.pth")))
            print('E')




"""
    def evaluate(self,env_test,ob_test,step_env,num_eval_episodes,action_repeat,writer):
        reward_returns = []
        cost_returns = []
        steps_until_dump_obs = 20
        log = {"step": [], "return": [], "cost": []}
        def coord_to_im_(coord):
            coord = (coord + 1.5) * 100
            return coord.astype(int)

        obs_list = []
        # track_list = []
        recons_list = []
        video_spf = 8 // action_repeat
        video_fps = 25 / video_spf
        # render_kwargs = deepcopy(self.env_test.env._render_kwargs["pixels"])
        # render_kwargs["camera_name"] = "track"
        for i in range(num_eval_episodes):
            self.z1 = None
            self.z2 = None

            # self.env_test.unwrapped.sim.render_contexts[0].vopt.geomgroup[:] = 1 # render all objects, including hazards
            state, ometer, tgt_state = env_test.reset.remote()#self.env_test.reset()

            # self.env_test.unwrapped.sim.render_contexts[0].vopt.geomgroup[:] = 1 # render all objects, including hazards
            ob_test.reset_episode(state, ometer, tgt_state)

            # self.env_test.unwrapped.sim.render_contexts[0].vopt.geomgroup[:] = 1 # render all objects, including hazards
            episode_return = 0.0
            cost_return = 0.0
            done = False
            eval_step = 0
            while not done:

                action = self.explore(ob_test)
                if i == 0 and eval_step % video_spf == 0:
                    im = ob_test.tgt_state[0][-1].astype("uint8")
                    obs_list.append(im)
                    reconstruction = sample_reproduction(self.latent, self.device, ob_test.state, ob_test.ometer,np.array([ob_test._action]))[0][-1] * 255
                    reconstruction = reconstruction.astype("uint8")
                    recons_list.append(reconstruction)

                    # track = self.env_test.unwrapped.sim.render(**render_kwargs)[::-1, :, :]
                    # track = np.moveaxis(track,-1,0)
                    # track_list.append(track)
                if steps_until_dump_obs == 0:
                    writer.add_image(f"observation/eval_state",ob_test.tgt_state[0][-1].astype(np.uint8),global_step=step_env)
                    #self.debug_save_obs(self.ob_test.tgt_state[0][-1], "eval_state", step_env)

                    reconstruction = sample_reproduction(self.latent, self.device, ob_test.state, ob_test.ometer,np.array([ob_test._action]))[0][-1] * 255
                    writer.add_image(f"observation/eval_reconstruction", reconstruction.astype(np.uint8),
                                     global_step=step_env)


                steps_until_dump_obs -= 1

                # self.env_test.env.env.sim.render_contexts[0].vopt.geomgroup[:] = 1 # render all objects, including hazards
                state, ometer, tgt_state, reward, done, info = env_test.step.remote()#self.env_test.step(action)
                # self.env_test.env.env.sim.render_contexts[0].vopt.geomgroup[:] = 1 # render all objects, including hazards
                cost = info["cost"]

                ob_test.append(state, ometer, tgt_state, action)
                episode_return += reward
                cost_return += cost

                eval_step += 1
            if i == 0:
                print("SAVE video in writer")
                writer.add_video(f"vid/eval", [np.concatenate([obs_list, recons_list], axis=3)],global_step=step_env, fps=video_fps)

            reward_returns.append(episode_return)
            cost_returns.append(cost_return)
        self.z1 = None
        self.z2 = None


        log["step"].append(step_env)
        mean_reward_return = np.mean(reward_returns)
        mean_cost_return = np.mean(cost_returns)
        median_reward_return = np.median(reward_returns)
        median_cost_return = np.median(cost_return)
        log["return"].append(mean_reward_return)
        log["cost"].append(mean_cost_return)

        # # Log to TensorBoard.
        # writer.add_scalar("return/test", mean_reward_return, step_env)
        # writer.add_scalar("return/test_median", median_reward_return, step_env)
        # writer.add_scalar("cost/test", mean_cost_return, step_env)
        # writer.add_scalar("cost/test_median", median_cost_return, step_env)
        # writer.add_histogram("return/test_hist", np.array(reward_returns), step_env)
        # writer.add_histogram("cost/test_hist", np.array(cost_returns), step_env)
        #
        # print(
        #     f"Steps: {step_env:<6}   " f"Return: {mean_reward_return:<5.1f} " f"CostRet: {mean_cost_return:<5.1f}   " f"Time: {str(timedelta(seconds=int(time() - start_time)))}")
        #

        # Return the evaluation results
        return {
            'mean_reward_return': mean_reward_return,
            'mean_cost_return': mean_cost_return,
            'median_reward_return': median_reward_return,
            'median_cost_return': median_cost_return,
            'reward_returns': np.array(reward_returns),
            'cost_returns': np.array(cost_returns),
            'log': log
        }

    def step(self, env,ob, t, is_pid, buffer,writer=None):

        t += 1

        if is_pid:
            action = env.pid_sample.remote()#self.env.pid_sample()



        else:
            action = self.explore(ob)
        # env.env.env.sim.render_contexts[0].vopt.geomgroup[:] = 1 # render all objects, including hazards
        state, ometer, tgt_state, reward, done, info = env.step.remote()#self.env.step(action)

        env.display.remote()#self.env.display()

        cost = info["cost"]
        self.lastcost = cost
        self.episode_costreturn += cost
        self.episode_rewardreturn += reward


        mask = False if t >= 1000 else done # 1000 : self.env._max_episode_steps


        ob.append(state, ometer, tgt_state, action)

        buffer.append.remote(action, reward, mask, state, ometer, tgt_state, done, cost)

        if done:
            if not is_pid:
                self.last_costreturn = self.episode_costreturn

                self.epoch_costreturns.append(self.episode_costreturn)
                self.epoch_rewardreturns.append(self.episode_rewardreturn)
            self.episode_costreturn = 0
            self.episode_rewardreturn = 0
            t = 0
            state, ometer, tgt_state = env.reset.remote()#self.env.reset()
            ob.reset_episode(state, ometer, tgt_state)
            buffer.reset_episode.remote(state, ometer, tgt_state)
            self.z1 = None
            self.z2 = None
        return t

"""