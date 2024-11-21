from gym.envs.registration import register

register(
    id='CarlaRlEnv-v0',
    entry_point='carla_rl_env.carla_env:CarlaRlEnv',
)