IS_OFFICE = False

default_cfg = {
    'IS_OFFICE': IS_OFFICE,
    'batch_size': 8 if not IS_OFFICE else 2,
    'max_epochs': 1000 if not IS_OFFICE else 3,
    'update_checkpoint_frequency': 20 if not IS_OFFICE else 2, # best checkpoint is saved independently
    'lr': 1e-5,
    'early_stop_patience': 100,
    'early_stop_min_delta': 0.01,
    'model_type': 'nvidia/mit-b2',
    'checkpoint_load_path': False,
}
