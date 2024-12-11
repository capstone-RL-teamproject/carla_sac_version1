import argparse
import os
import ray
from ray.rllib.agents.ppo import PPOTrainer, PPOConfig
from gym.envs.registration import register
import gym
from WrappedGymEnv import WrappedGymEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=2 * 10**6)
    parser.add_argument("--domain_name", type=str, default="CarlaRlEnv-v0")
    parser.add_argument("--task_name", type=str, default="run")
    parser.add_argument("--action_repeat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_false")
    args = parser.parse_args()

    params = {
        "carla_port": 2000,
        "map_name": "Town10HD",
        "window_resolution": [1080, 1080],
        "grid_size": [3, 3],
        "sync": True,
        "no_render": True,
        "display_sensor": False,
        "ego_filter": "vehicle.tesla.model3",
        "num_vehicles": 50,
        "num_pedestrians": 20,
        "enable_route_planner": True,
        "sensors_to_amount": [
            "left_rgb",
            "front_rgb",
            "right_rgb",
            "top_rgb",
            "lidar",
            "radar",
        ],
        "image_size": 64,
        "action_repeat": args.action_repeat,
    }

    register(
        id="CarlaRlEnvRaw-v0",
        entry_point="carla_rl_env.carla_env:CarlaRlEnv",
    )

    def env_creator(env_config):
        env = gym.make("CarlaRlEnvRaw-v0", params=env_config["params"])
        env = WrappedGymEnv(
            env,
            image_size=env_config["params"]["image_size"],
            action_repeat=env_config["params"]["action_repeat"],
        )
        return env

    register(
        id=args.domain_name,
        entry_point=env_creator,
    )

    ray.init(ignore_reinit_error=True)

    config = (
        PPOConfig()
        .environment(
            env=args.domain_name,
            env_config={"params": params},
            disable_env_checking=True,
        )
        .framework("torch")
        .rollouts(num_rollout_workers=1, rollouts_per_iteration=1)
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=8192,
            sgd_minibatch_size=256,
            num_sgd_iter=10,
            clip_param=0.2,
        )
        .resources(num_gpus=1 if args.cuda else 0)
        .seed(args.seed)
    )

    trainer = PPOTrainer(config=config)

    total_timesteps = 0
    target_timesteps = args.num_steps
    eval_interval = 50000

    while total_timesteps < target_timesteps:
        result = trainer.train()
        total_timesteps = result["timesteps_total"]
        print(
            f"Iter: {result['training_iteration']}, "
            f"Timesteps: {total_timesteps}, "
            f"Episode Reward Mean: {result['episode_reward_mean']}, "
            f"Episode Len Mean: {result['episode_len_mean']}"
        )
        if total_timesteps % eval_interval == 0:
            checkpoint_dir = trainer.save()
            print(f"Checkpoint saved at {checkpoint_dir}")
            eval_results = trainer.evaluate()
            print("Evaluation results:", eval_results)

    ray.shutdown()
