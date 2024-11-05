import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import argparse


def env_creator(env_config):
    return CarlaEnvRllib(env_config)


def main(args):
    ray.init()

    # 환경 등록
    register_env("carla_env", env_creator)

    # PPO 설정
    config = (
        PPOConfig()
        .environment(
            "carla_env",
            env_config={
                "carla_port": args.carla_port,
                "map_name": args.map_name,
                "window_resolution": [1080, 1080],
                "grid_size": [3, 3],
                "sync": True,
                "no_render": not args.render,
                "display_sensor": args.render,
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
            },
        )
        .framework("torch")
        .training(
            model={
                "conv_filters": [
                    [16, [8, 8], 4],
                    [32, [4, 4], 2],
                    [256, [11, 11], 1],
                ],
                "post_fcnet_hiddens": [256, 256],
                "custom_model_config": {},
            },
            gamma=0.99,
            lambda_=0.95,
            kl_coeff=0.5,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=30,
        )
        .resources(num_gpus=args.num_gpus)
        .rollouts(num_rollout_workers=args.num_workers, rollout_fragment_length=200)
    )

    # 학습 실행
    tune.run(
        "PPO",
        config=config.to_dict(),
        stop={"training_iteration": args.num_iterations},
        checkpoint_freq=10,
        local_dir="./logs",
        name="carla_ppo",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iterations", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--map-name", type=str, default="Town10HD")
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()
    main(args)
