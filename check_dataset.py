import argparse

from PIL import Image
from transformers import SegformerImageProcessor

from configs.train_cfg import default_cfg
from dataset.dataset_utils import check_dataset, clean_dataset
from dataset.disaster_dataset import DisasterSegDataset
from utils.common import common_paths


parser = argparse.ArgumentParser()
parser.add_argument('--delete', action='store_true', help='By default the filenames are only printed. Using --delete will also remove the files.')
parser.add_argument("--check", action='store_true', help='If provided, the dataset is scanned and corrupted filenames are stored in a .txt file.')
args = parser.parse_args()

assert not (args.check and args.delete), 'Only one of the provided args can be True: --delete --check'

if not args.check:
    clean_dataset(errorfile='errors.txt', dry=not args.delete)
else:
    cfg = default_cfg

    # Dataset
    preprocessor = SegformerImageProcessor()

    train_dataset = DisasterSegDataset(
        root_dir=common_paths["dataset_root"],
        preprocessor=preprocessor,
        train=True,
        augment_data=False,
    )

    valid_dataset = DisasterSegDataset(
        root_dir=common_paths["dataset_root"],
        preprocessor=preprocessor,
        train=False,
        augment_data=False
    )

    check_dataset(train_dataset)
    check_dataset(valid_dataset)