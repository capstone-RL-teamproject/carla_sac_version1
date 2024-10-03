import torch
import torch.nn as nn
import copy
import numpy as np
import torch.nn.functional as F


import torch.optim as optim
from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path

from torch.distributions import Normal







class ConvLayer(nn.Module):
    def __init__(self, conv1_filter=8, conv1_kernel=4, conv1_stride=2, conv1_padding=1,
                 conv2_filter=16, conv2_kernel=3, conv2_stride=2, conv2_padding=1,
                 conv3_filter=32, conv3_kernel=3, conv3_stride=2, conv3_padding=1,
                 conv4_filter=64, conv4_kernel=3, conv4_stride=2, conv4_padding=1,
                 conv5_filter=128, conv5_kernel=3, conv5_stride=1, conv5_padding=1,
                 conv6_filter=256, conv6_kernel=3, conv6_stride=1, conv6_padding=1):
        super(ConvLayer, self).__init__()

        # if c+l : 6c, bev : 3

        self.conv1 = nn.Conv2d(3, conv1_filter, conv1_kernel, conv1_stride, conv1_padding)
        self.conv2 = nn.Conv2d(conv1_filter, conv2_filter, conv2_kernel, conv2_stride, conv2_padding)
        self.conv3 = nn.Conv2d(conv2_filter, conv3_filter, conv3_kernel, conv3_stride, conv3_padding)
        self.conv4 = nn.Conv2d(conv3_filter, conv4_filter, conv4_kernel, conv4_stride, conv4_padding)
        self.conv5 = nn.Conv2d(conv4_filter, conv5_filter, conv5_kernel, conv5_stride, conv5_padding)
        self.conv6 = nn.Conv2d(conv5_filter, conv6_filter, conv6_kernel, conv6_stride, conv6_padding)
        self.flat = nn.Flatten()

    def forward(self, observation):
        conv1_output = F.leaky_relu(self.conv1(observation))
        conv2_output = F.leaky_relu(self.conv2(conv1_output))
        conv3_output = F.leaky_relu(self.conv3(conv2_output))
        conv4_output = F.leaky_relu(self.conv4(conv3_output))
        conv5_output = F.leaky_relu(self.conv5(conv4_output))
        conv6_output = F.leaky_relu(self.conv6(conv5_output))
        output = self.flat(conv6_output)
        return output

# policy network
class Actor(nn.Module):
    def __init__(self, action_scale,action_bias,action_shape=(2,)):

        super(Actor,self).__init__()


        self.action_dim = action_shape[0]


        self.action_scale = torch.tensor(action_scale,dtype=torch.float32).to('cuda:0')
        self.action_bias = torch.tensor(action_bias,dtype=torch.float32).to('cuda:0')

        self.conv = ConvLayer()
        self.fc1 = nn.Linear(4096,512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256,128)

        self.mu = nn.Linear(128,self.action_dim)
        self.log_std = nn.Linear(128,self.action_dim)

    def forward(self,state):
        conv_output = self.conv(state)
        #print(f"\n\nconv output: {conv_output.shape}\n\n")
        x = F.relu(self.fc1(conv_output))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        mu = self.mu(x)
        log_std = self.log_std(x)

        log_std = torch.tanh(log_std)  # -1~1 사이로 제한


        return mu,log_std

    def get_action(self,state):
        """

        :param state:
        :return: the action based on a squashed gaussian policy
        """

        mean,log_std = self.forward(state)

        std = log_std.exp() # log 통해 std 계산하기 때문이다

        normal = Normal(mean,std) # mu(s), std(s) 따르는 정규 분포 만든다 for reparameterization trick

        x_t = normal.rsample() # reparameterization trick => (mean + std * N(0,1)) ==> Action
        y_t = torch.tanh(x_t)

        # scaling & bias
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t) # sampling한 action의 확률값을 로그로 계산한다
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6) # tanh로 변형된값을 역함수를 통해 복원해야 한다 1-tanh^2을 하는 과정
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action,log_prob,mean

class Critic(nn.Module):
    def __init__(self):
        super(Critic,self).__init__()
        self.conv = ConvLayer()


        # Q1 architecture
        self.l1 = nn.Linear(4098, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128,1)




    def forward(self, state, action):
        #print(f"critic debug : state shape {state.shape}\n action shape {action.shape}\n ")
        conv_output = self.conv(state)
        #print(f"conv_output shape {conv_output.shape}\n")

        x = torch.cat([conv_output, action], 1)

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))

        x = self.l4(x)

        return x



class SAC(object):
    def __init__(self,opt,device):

        self.action_shape = opt.action_shape
        self.action_scale = opt.action_scale
        self.action_bias = opt.action_bias
        self.device = device


        self.writer = opt.write

        self.tau = opt.tau
        self.gamma = opt.gamma
        self.actor_lr = opt.lr_init
        self.critic_lr = opt.lr_init
        self.batch_size = opt.batch_size
        self.total_it = 0
        self.policy_freq = opt.policy_freq

        # automatic entropy coefficient alpha -> but high variance
        self.target_entropy = -torch.Tensor(self.action_shape).to(self.device)

        self.log_alpha = torch.zeros(1, requires_grad=True,device=self.device)
        self.autotune_alpha = self.log_alpha.exp().item()
        self.autotune_alpha_optimizer = optim.AdamW(params=[self.log_alpha], lr=self.critic_lr)

        # network
        self.actor = Actor(self.action_scale,self.action_bias).to(self.device)

        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=self.actor_lr)

        self.critic = Critic().to(self.device)
        self.critic_target = copy.deepcopy(self.critic).eval()
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.critic_lr)

        self.critic2 = Critic().to(self.device)
        self.critic_target2 = copy.deepcopy(self.critic2).eval()
        self.critic_optimizer2 = optim.AdamW(self.critic2.parameters(), lr=self.critic_lr)


    def select_action(self,state,random_sample=False):
        state = torch.tensor(state,dtype=torch.uint8, device=self.device).float().div_(255.0)
        state = state.unsqueeze_(0)

        if random_sample:
            # carla env main random policy
            action = np.random.normal([1.0, 0.0],[0.3, 0.3],self.action_shape)
            action = np.clip(action, -np.ones(2),np.ones(2))

        else:

            action, _, _ = self.actor.get_action(state)
            action = action.detach().cpu().numpy()


        action = action.flatten()


        return action

    def has_enough_experience(self,buffer)->bool:
        """True if buffer hase enough experience to train """

        return len(buffer) >= buffer.batch_size

    def train(self,beta,alpha,PER_buffer):
        """
            with PER Buffer + per alpha apply adavtive schedule -> my experiment
            : td mean, td std 비율에 따라 alpha 동적으로 변화한다  -> utills에 schedule추가
            Critic_loss = MSE(Q, Q_target)
            Actor_loss = α * log_pi(a|s) - Q(s,a)
            where:
                actor_target(state) -> action
                critic_target(state, action) -> Q-value

        :param PER_buffer:
        :return:
        """

        idxs, experiences, sampling_weights = PER_buffer.sample(beta,alpha)

        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])

        states = torch.Tensor(states).to(self.device)
        actions = torch.Tensor(actions).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)
        dones = torch.Tensor(dones).to(self.device)


        with torch.no_grad():

            #current policy에서 next action가져 온다
            next_action,next_log_pi,_ = self.actor.get_action(next_states)

            Q_target1 = self.critic_target(next_states,next_action)
            Q_target2 = self.critic_target2(next_states, next_action)

            # maxmize object function
            target_Q = torch.min(Q_target1, Q_target2) - self.autotune_alpha * next_log_pi

            # Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
            target_Q = rewards.flatten()+((1-dones.flatten())*self.gamma * target_Q.view(-1))

        cur_Q1 = self.critic(states,actions).view(-1)
        cur_Q2 = self.critic2(states,actions).view(-1)

        TD_error1 = target_Q - cur_Q1
        TD_error2 = target_Q - cur_Q2
        # priority 선택시 overestimate 최소 위해 최소값을 선택하였는데 평균값으로 실험해봐도 괜찮을것 같다

        TD_Error = torch.min(TD_error1, TD_error2)
        priority = (TD_Error.abs().cpu().detach().numpy().flatten())
        PER_buffer.update_priorities(idxs, priority + 1e-6)  # priority must positiv

        _sampling_weights = (torch.Tensor(sampling_weights).view((-1, 1))).to(self.device)

        # MSE loss with importance sampling
        critic_loss1 = torch.mean(torch.square(_sampling_weights * TD_error1))
        critic_loss2 = torch.mean(torch.square(_sampling_weights * TD_error2))

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer.step()

        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()

        if self.total_it % self.policy_freq == 0:
            for _ in range(self.policy_freq):
                action_pi,log_pi,_ = self.actor.get_action(states)
                q1 = self.critic(states,action_pi)
                q2 = self.critic2(states,action_pi)
                q_pi = torch.min(q1,q2)

                actor_loss = (((self.autotune_alpha * log_pi) - q_pi)*_sampling_weights).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # autoutune alpha
                with torch.no_grad():
                    _,log_pi,_ = self.actor.get_action(states)

                autotune_alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                self.autotune_alpha_optimizer.zero_grad()
                autotune_alpha_loss.backward()
                self.autotune_alpha_optimizer.step()
                self.autotune_alpha = self.log_alpha.exp().item()

            self.soft_update(self.critic, self.critic_target, self.tau)
            self.soft_update(self.critic2, self.critic_target2, self.tau)

            self.total_it += 1










    def soft_update(self, network, target_network, rate):
        for network_params, target_network_params in zip(network.parameters(), target_network.parameters()):
            target_network_params.data.copy_(target_network_params.data * (1.0 - rate) + network_params.data * rate)

    def save(self,filename):

        # save model
        torch.save(self.actor.state_dict(), f"{filename}+_actor")
        torch.save(self.critic.state_dict(), f"{filename}+_critic_1")
        torch.save(self.critic2.state_dict(), f"{filename}+_critic_2")


    def load(self, filename):
        # load model
        self.actor.load_state_dict(torch.load(f"{filename}+_actor"))
        self.actor_target = copy.deepcopy(self.actor)

        self.critic.load_state_dict(torch.load(f"{filename}+_critic_1"))
        self.critic_target = copy.deepcopy(self.critic)

        self.critic2.load_state_dict(torch.load(f"{filename}+_critic_2"))
        self.critic_target = copy.deepcopy(self.critic2)
