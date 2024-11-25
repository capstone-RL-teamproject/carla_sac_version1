import os
import argparse
from safe_slace.trainer import train_ppo
import ray


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="ppo_carla")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=200)
    args = parser.parse_args()

    # Ray 초기화
    ray.init()

    # 로그 디렉토리 생성
    log_dir = os.path.join("logs", args.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    # 학습 설정
    config = {
        "env": "CarlaEnv",
        "num_workers": 4,
        "num_episodes": args.num_episodes,
        "seed": args.seed,
        "log_dir": log_dir,
    }

    # PPO 학습 시작
    trainer = train_ppo(config)

    # 최종 모델 저장
    final_checkpoint = trainer.save(log_dir)
    print(f"Final model saved at: {final_checkpoint}")

    ray.shutdown()


if __name__ == "__main__":
    main()
