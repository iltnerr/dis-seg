IS_OFFICE = True

default_cfg = {
    'IS_OFFICE': IS_OFFICE,
    'model_type': 'nvidia/mit-b2',
    'checkpoint_load_path': '/cdtemp/richard/coding/model-checkpoints/dis-seg/sf-b2.pt',
    'batch_size': 8 if not IS_OFFICE else 2,
}