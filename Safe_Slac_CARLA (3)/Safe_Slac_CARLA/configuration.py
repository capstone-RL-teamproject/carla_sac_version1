def get_default_config():
    cfg = {
            "domain_name": "Safe_SLAC_CARLA",
            "task_name": "run",
            "action_repeat": 4,
            "image_size": 64, #64
            "image_noise": 0.4,
            "seed": 0,
            "num_sequences": 10,
            "gamma_c": 0.995,
            "buffer_size": int(1e6),
            "feature_dim": 200,
            "z2_dim": 200,
            "hidden_units": (256, 256),
            "batch_size_latent": 64, #32
            "batch_size_sac":32, #64
            "lr_sac": 2e-4, #2e-4,
            "lr_latent": 1e-4, #1e-4,
            "start_alpha": 4e-3,
            "start_lagrange": 0.02,
            "num_steps": 3e6, #2e6,
            "initial_learning_steps": 400, #30_000,
            "initial_collection_steps": 1000,#3_000, #30_000,
            "collect_with_policy": False,
            "eval_interval": 20_000, #25*10**3,
            "num_eval_episodes": 5,#10,
            "grad_clip_norm": 40.0,
            "tau": 5e-3,
            "train_steps_per_iter": 250,#500,#100,
            "env_steps_per_train_step": 100
        }

    return cfg