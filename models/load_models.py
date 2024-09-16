import torch

from utils.common import common_paths
from transformers import SegformerForSemanticSegmentation


def load_segformer(id2label, label2id, checkpoint_file=None):
    config_path = common_paths['segformer_config_path']
    SFConfiguration = SegformerForSemanticSegmentation.config_class.from_pretrained(config_path, num_labels=len(id2label), id2label=id2label, label2id=label2id)
    model = SegformerForSemanticSegmentation(SFConfiguration)
    
    if checkpoint_file is not None:
        state_dict = torch.load(checkpoint_file, map_location="cpu")
        loaded_state_dict_keys = list(state_dict.keys())

        # _load_pretrained_model() returns tuple with additional vars. Only need the model, which is index 0.
        model = SegformerForSemanticSegmentation._load_pretrained_model(model, state_dict, loaded_state_dict_keys, checkpoint_file, config_path)[0]

    return model