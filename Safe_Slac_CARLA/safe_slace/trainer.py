import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn


# 커스텀 신경망 모델
class CarlaCNNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # CNN 레이어
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 특징 결합 레이어
        self.combined_layer = nn.Sequential(
            nn.Linear(3136 + 4, 512),  # 3136은 CNN 출력, 4는 속도+위치
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # 정책 및 가치 헤드
        self.policy = nn.Linear(256, num_outputs)
        self.value = nn.Linear(256, 1)

    def forward(self, input_dict, state, seq_lens):
        # 이미지 처리
        obs = input_dict["obs"]
        img = obs["camera"].float() / 255.0
        img = img.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        img_features = self.cnn(img)

        # 다른 관측치들과 결합
        other_features = torch.cat([obs["speed"], obs["location"]], dim=1)

        combined = torch.cat([img_features, other_features], dim=1)
        features = self.combined_layer(combined)

        # 정책 출력
        policy = self.policy(features)

        return policy, state

    def value_function(self):
        return self.value(self._features).squeeze(1)


def train_ppo(config=None):
    if config is None:
        config = {
            "env": "CarlaEnv",
            "num_workers": 4,
            "train_batch_size": 4000,
            "sgd_minibatch_size": 128,
            "num_sgd_iter": 30,
            "lr": 3e-4,
            "gamma": 0.99,
            "lambda": 0.95,
            "clip_param": 0.2,
            "model": {
                "custom_model": "carla_cnn_model",
            },
            "framework": "torch",
        }

    # 커스텀 모델 등록
    ModelCatalog.register_custom_model("carla_cnn_model", CarlaCNNModel)

    # PPO 트레이너 생성
    trainer = PPO(config=config)

    # 학습 실행
    for i in range(200):
        result = trainer.train()
        print(f"Episode {i}: reward = {result['episode_reward_mean']}")

        if i % 10 == 0:
            checkpoint_dir = trainer.save()
            print(f"Checkpoint saved at {checkpoint_dir}")

    return trainer
