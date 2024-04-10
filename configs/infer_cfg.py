is_office = True

default_cfg = {
    'batch_size': 8 if not is_office else 2,
    'checkpoint_load_path': '/cdtemp/richard/coding/model-checkpoints/dis-seg/sf-b2.pt',
    'is_office': is_office,
    'segformer_config_path': './models/segformer-b2-config.json',
}