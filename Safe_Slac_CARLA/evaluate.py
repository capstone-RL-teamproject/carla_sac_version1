import carla
import numpy as np
from safe_slace.trainer import PPO
import time


def run_autonomous_driving(start_point, end_point, model_path):
    # 환경 초기화
    env = CarlaEnv()

    # 학습된 모델 로드
    config = {
        "env": "CarlaEnv",
        "framework": "torch",
        "model": {
            "custom_model": "carla_cnn_model",
        },
    }
    trainer = PPO(config=config)
    trainer.restore(model_path)

    # 시작점으로 차량 이동
    obs, _ = env.reset(start_location=start_point)
    env.target_location = np.array([end_point.x, end_point.y, end_point.z])

    total_reward = 0
    done = False

    while not done:
        # 행동 예측 및 실행
        action = trainer.compute_single_action(obs)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

        # 상태 출력
        print(
            f"Speed: {obs['speed'][0]:.2f} m/s, "
            f"Distance to goal: {np.linalg.norm(obs['location'] - env.target_location):.2f} m, "
            f"Reward: {reward:.2f}"
        )

        if info.get("collision", False):
            print("충돌 발생!")
            break

        time.sleep(0.1)

    print(f"주행 완료! 총 보상: {total_reward:.2f}")
    env._clear_actors()


if __name__ == "__main__":
    # 시작점과 목적지 설정
    start_point = carla.Location(x=0, y=0, z=0)
    end_point = carla.Location(x=100, y=100, z=0)
    model_path = "logs/ppo_carla/checkpoint_000200/checkpoint-200"

    run_autonomous_driving(start_point, end_point, model_path)
