import ray
from ray.train.torch import TorchTrainer,TorchConfig

from ray.train import ScalingConfig,RunConfig, FailureConfig
import torch
import argparse
import os
from datetime import datetime

from safe_slace.Distributed_trainer import train_loop
import numpy as np
from configuration import get_default_config
from gym.envs.registration import register
from safe_slace.ReplayBuffer import CostReplayBuffer
from gym.spaces.box import Box
from safe_slace.algo import LatentPolicySafetyCriticSlac
from WrappedGymEnv import WrappedGymEnv
import gym
from ray.util.placement_group import placement_group
import json

from ray.tune.logger import TBXLoggerCallback
from ray.air import RunConfig
def main(args):

    os.environ['MASTER_ADDR'] = '127.0.0.1'  # 혹은 네트워크 내에서 사용 가능한 IP
    os.environ['MASTER_PORT'] = '29500'
    os.environ["RAY_DEDUP_LOGS"] = "0"
    os.environ['RAY_memory_usage_threshold'] = '0.99'
    os.environ["RAY_memory_monitor_refresh_ms"] = "0"

    # ray.init(
    #     _system_config={
    #         # Allow spilling until the local disk is 99% utilized.
    #         # This only affects spilling to the local file system.
    #         "local_fs_capacity_threshold": 0.99,
    #         "object_spilling_config": json.dumps(
    #             {
    #                 "type": "filesystem",
    #                 "params": {
    #                     "directory_path": "/tmp/spill",
    #                 }
    #             },
    #         )
    #     },object_store_memory=31 * 1024 * 1024 * 1024,ignore_reinit_error=True
    # )
    ray.init(address="auto",ignore_reinit_error=True)#address="auto",ignore_reinit_error=True




    config = get_default_config()
    config["domain_name"] = args.domain_name
    config["task_name"] = args.task_name
    config["seed"] = args.seed
    config["num_steps"] = args.num_steps


    training_config = {
        "num_sequences": config["num_sequences"],
        "gamma_c": config["gamma_c"],
        "action_repeat": config["action_repeat"],
        "device": torch.device("cuda" if args.cuda else "cpu"),
        "seed": config["seed"],
        "buffer_size": config["buffer_size"],
        "feature_dim": config["feature_dim"],
        "z2_dim": config["z2_dim"],
        "hidden_units": config["hidden_units"],
        "batch_size_latent": config["batch_size_latent"],
        "batch_size_sac": config["batch_size_sac"],
        "lr_sac": config["lr_sac"],
        "lr_latent": config["lr_latent"],
        "start_alpha": config["start_alpha"],
        "start_lagrange": config["start_lagrange"],
        "grad_clip_norm": config["grad_clip_norm"],
        "tau": config["tau"],
        "image_noise": config["image_noise"],
        "initial_learning_steps":config["initial_learning_steps"],
        "initial_collection_steps":config["initial_collection_steps"],
        "collect_with_policy" : config["collect_with_policy"],
        "eval_interval" : config["eval_interval"],
        "num_eval_episodes" : config["num_eval_episodes"],
        "train_steps_per_iter" : config["train_steps_per_iter"],
        "env_steps_per_train_step" : config["env_steps_per_train_step"],
        "num_steps":config["num_steps"],
        "image_size":config["image_size"]



    }





    log_dir = os.path.join(
        "logs",
        f"{config['domain_name']}-{config['task_name']}",
        f'slac-seed{config["seed"]}-{datetime.now().strftime("%Y%m%d-%H%M")}',
    )

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # model 경로 반드시 절대 경로로 설정하기!
    model_dir = os.path.abspath("/home/ad13/meang_RL/Safe_Slac_CARLA/models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)




    training_loop_config={
        "log_dir":log_dir,
        "model_dir":model_dir,
        "training_config":training_config

    }


    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,
        resources_per_worker={"CPU":10,"GPU":0.625},
        placement_strategy="PACK"

    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        train_loop_config=training_loop_config,
        scaling_config=scaling_config,
        run_config=RunConfig(callbacks=[TBXLoggerCallback()])


    )
    trainer.fit()
    ray.shutdown()






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=2 * 10 ** 6)
    parser.add_argument("--domain_name", type=str, default="CarlaRlEnv-v0")
    parser.add_argument("--task_name", type=str, default="run")
    parser.add_argument("--action_repeat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_worker", type=int, default=1)

    parser.add_argument("--cuda", action="store_false")
    args = parser.parse_args()
    main(args)