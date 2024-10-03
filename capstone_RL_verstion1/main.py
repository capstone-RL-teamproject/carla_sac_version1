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

FLAG=True# 삭제해야하는 코드

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("\nDevice is ",device)
print()

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


    parser.add_argument("--eval_freq", default=1e3, type=int)  # How often (time steps) we evaluate 1e3

    parser.add_argument('--Loadmodel', type=bool, default=False,help='Load pretrained model or Not')  # 훈련 마치고 나서는 True로 설정 하기

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
        'sensors_to_amount': ['front_rgb', 'lidar'],
    }

    env= WrappedGymEnv(gym.make("CarlaRlEnv-v0",params=params),args)

    writer = SummaryWriter(log_dir=f"./results/{file_name}")
    args.summary_writer = writer


    args.action_shape = env.action_space.shape[0]
    args.action_scale = env.action_scale
    args.action_bias = env.action_bias


    # Set seeds
    #env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(args)

    policy = SAC(args,device)

    buffer = PER_Buffer(args, device=device)

    BETA = args.beta_init
    BUFFER_ALPHA = 0.6

    beta_scheduler = LinearSchedule(args.beta_gain_steps,args.beta_init,1.0) # beta end is 1.0 : scheduler must go to 1.0
    buffer_alpha_scheduler = adaptiveSchedule(args.alpha_max,args.alpha_min)

    actor_lr_scheduler = LinearSchedule(args.lr_decay_steps, args.lr_init, args.lr_end)
    critic_lr_scheduler = LinearSchedule(args.lr_decay_steps, args.lr_init, args.lr_end)

    # load the model
    if args.Loadmodel:
        policy.load(f"./models/{file_name}")

    # test
    if args.render and args.Loadmodel:
        pass

    else:
        for t in range(int(args.max_timesteps)):


            state = env.reset()



            done = False
            episode_reward = 0
            episode_cost=0


            while True:

                if not args.Loadmodel and t < args.start_timesteps:
                    action = policy.select_action(state, random_sample=True)


                else:

                    action = policy.select_action(state, random_sample=False)
                    #print(f"select action: {action} and shape is {action.shape}\n\n")

                # Perform action
                next_state, reward, done, info = env.step(action)
                env.display()
                cost = info['cost'] # collision & invasion cost

                experience = Experience(state, action, reward, next_state, done)

                buffer.add(experience)

                if done:
                    print(f"done after {t+1} steps done is {done}")

                    break

                # train      and t > args.start_timesteps
                if policy.has_enough_experience(buffer) and t > args.start_timesteps:
                    #print(f"\ntimestep {t} train start\n")
                    policy.train(BETA, BUFFER_ALPHA, buffer)
                    #print(f"\ntrain debug {c1} ,{c2} ,{actor_loss} ,{alpha}, {alpha_loss}\n\n")




                    BETA = beta_scheduler.value(t)
                    td_mean = np.mean(buffer.buffer[:len(buffer)]["priority"])
                    td_std = np.std(buffer.buffer[:len(buffer)]["priority"])
                    BUFFER_ALPHA = buffer_alpha_scheduler.value(td_mean, td_std)

                    # 삭제해야하는 코드
                    if FLAG:
                        print(f"\ntimestep {t} train start beta {BETA} td_mean {td_mean} td std {td_std} buffer alpha {BUFFER_ALPHA}\n")

                    #print(f"\n scheduler debug {BETA} ,{td_mean} ,{td_std}, {ALPHA}\n\n")

                    # actor lr scheduler
                    for p in policy.actor_optimizer.param_groups:

                        p['lr'] = actor_lr_scheduler.value(t)

                    # critic lr scheduler
                    for p in policy.critic_optimizer.param_groups:
                        p['lr'] = critic_lr_scheduler.value(t)

                    for p in policy.critic_optimizer2.param_groups:
                        p['lr'] = critic_lr_scheduler.value(t)

                FLAG = False # 삭제해야하는 코드
                state = next_state
                episode_reward += reward
                episode_cost+=cost

                # Evaluate episode
                if (t + 1) % args.eval_freq == 0:
                    print("\nEvaluate score\n")
                    avg_reward,avg_cost = eval_policy(policy, env)
                    if args.write:
                        args.summary_writer.add_scalar('eval reward', avg_reward, global_step=t)
                        args.summary_writer.add_scalar('episode_reward', episode_reward, global_step=t)
                        args.summary_writer.add_scalar('eval_cost', avg_cost, global_step=t)
                        args.summary_writer.add_scalar('episode_cost', episode_cost, global_step=t)

                        args.summary_writer.add_scalar('BUFFER_alpha', BUFFER_ALPHA, global_step=t)
                        #args.summary_writer.add_scalar('critic_loss1', c1, global_step=t)
                        #args.summary_writer.add_scalar('critic_loss2', c2, global_step=t)
                        #args.summary_writer.add_scalar('actor_loss', actor_loss, global_step=t)
                        #args.summary_writer.add_scalar('autotune_alpha', alpha, global_step=t)
                        #args.summary_writer.add_scalar('autotune_alpha_loss', alpha_loss, global_step=t)

                        args.summary_writer.add_scalar('beta', BETA, global_step=t)

                    policy.save(f"./models/{file_name}")
                    print('writer add scalar and save model   ', 'steps: {}k'.format(int(t / 1000)), 'AVG reward:',int(avg_reward),'AVG cost:',int(avg_cost))
                    t = t + 1

        print(f"\n--------timestep : {t} reward : {episode_reward}  cost : {episode_cost}--------\n")






if __name__ == '__main__':
    main()


