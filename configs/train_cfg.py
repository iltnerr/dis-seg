is_office = False

default_cfg = {
    'batch_size': 8 if not is_office else 2,
    'early_stop_min_delta': 0.01,
    'early_stop_patience': 20,
    'is_office': is_office,
    'lr': 1e-5,
    'max_epochs': 200 if not is_office else 3,
    'segformer_config_path': './models/segformer-b2-config.json',
    'segformer_pretrained_weights_path': '/cdtemp/richard/coding/model-checkpoints/dis-seg/segformer-mit-b2-original-checkpoints.bin', # initialize pretrained weights for training
    'update_checkpoint_frequency': 20 if not is_office else 2, # best checkpoint is saved independently of this setting
}
