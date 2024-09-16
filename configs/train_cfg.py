from utils.common import office_hostname
import socket 


is_office = socket.gethostname() == office_hostname

default_cfg = {
    'batch_size': 8 if not is_office else 2,
    'early_stop_min_delta': 0.005,
    'early_stop_patience': 200,
    'expand_pytorch_alloc_mem': True, # if True, reduce reserved gpu memory by pytorch to avoid CUDA OOM errors. Default: False
    'is_office': is_office,
    'lr': 6e-5, # 6e-5 reported as initial lr for poly schedule 
    'max_epochs': 100 if not is_office else 3,
    'update_checkpoint_frequency': 20 if not is_office else 2, # best checkpoint is saved independently of this setting
    'use_augmentation': False,
    'warmup_interval': 5,
}
