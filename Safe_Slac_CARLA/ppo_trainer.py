import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn


class CarlaCNNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # CNN 네트워크 구성
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 측정값 처리를 위한 FC 레이어
        self.measurement_fc = nn.Linear(4, 64)

        # 통합 FC 레이어
        self.combined_fc = nn.Sequential(
            nn.Linear(64 * 64 + 64, 512), nn.ReLU(), nn.Linear(512, num_outputs)
        )

        # Value function
        self.value_fc = nn.Sequential(
            nn.Linear(64 * 64 + 64, 512), nn.ReLU(), nn.Linear(512, 1)
        )

    def forward(self, input_dict, state, seq_lens):
        # 이미지 처리
        obs = input_dict["obs"]
        img = obs["image"].float() / 255.0
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW
        img_features = self.cnn(img)

        # 측정값 처리
        measurements = obs["measurements"]
        measurement_features = self.measurement_fc(measurements)

        # 특징 결합
        combined = torch.cat([img_features, measurement_features], dim=1)

        # 정책 출력
        policy = self.combined_fc(combined)

        return policy, state

    def value_function(self):
        return self.value_fc(combined).squeeze(1)


def train_ppo():
    ray.init()

    # 커스텀 모델 등록
    ModelCatalog.register_custom_model("carla_cnn_model", CarlaCNNModel)

    config = (
        PPOConfig()
        .environment("CarlaEnv")
        .framework("torch")
        .training(
            model={
                "custom_model": "carla_cnn_model",
            },
            gamma=0.99,
            lr=3e-4,
            num_sgd_iter=10,
            train_batch_size=4000,
            num_workers=4,
        )
    )

    # 학습 실행
    results = tune.run(
        "PPO",
        config=config.to_dict(),
        stop={"training_iteration": 200},
        checkpoint_freq=10,
        checkpoint_at_end=True,
    )

    ray.shutdown()


if __name__ == "__main__":
    train_ppo()
