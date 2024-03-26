IS_OFFICE = True

default_cfg = {
    'IS_OFFICE': IS_OFFICE,
    'batch_size': 8 if not IS_OFFICE else 2,
    'num_epochs': 150 if not IS_OFFICE else 15,
    'update_frequency': 10 if not IS_OFFICE else 2, # safe checkpoints every x epochs (does not affect best checkpoint)
    'lr': 0.0000006,
    'model_type': 'nvidia/mit-b2',
    'checkpoint_load_path': False,
}
