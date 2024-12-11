import argparse
import ray
from ray.rllib.agents.ppo import PPOTrainer
import gym
from gym.envs.registration import register
from WrappedGymEnv import WrappedGymEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to PPO checkpoint"
    )
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
        "action_repeat": 4,
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
        id="CarlaRlEnv-v0",
        entry_point=env_creator,
    )

    ray.init(ignore_reinit_error=True)

    trainer = PPOTrainer(
        env="CarlaRlEnv-v0",
        config={
            "env_config": {"params": params},
            "num_workers": 0,
            "framework": "torch",
        },
    )
    trainer.restore(args.checkpoint)

    env = gym.make("CarlaRlEnv-v0", params=params)
    for ep in range(3):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = trainer.compute_single_action(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print(f"Episode {ep}: reward={total_reward}")

    ray.shutdown()
