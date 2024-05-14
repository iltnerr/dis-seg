import numpy as np
import os

from PIL import Image
from PIL.Image import fromarray
from tqdm import tqdm


cls_dict = {
    0: 'Background',
    1: 'Building',
    2: 'Road',
    3: 'Vehicle',
    4: 'Debris',
    5: 'Fire',
    6: 'Water',
    7: 'Animal',
    8: 'Sky',
    9: 'Smoke',
    10: 'Tree',
    11: 'Person',
}

def pixel_values_to_pil_image(pixel_values):
    np_arr = np.array(pixel_values)
    reordered = np.moveaxis(np_arr, 0, -1)
    rescaled = ((reordered - reordered.min()) * (1/(reordered.max() - reordered.min()) * 255)).astype('uint8')
    return fromarray(rescaled)

def check_dataset(ds, log_errors=False, try_main_fn=False):
    """
    Check dataset for errors after annotation with gsam. 
    """
    pbar = tqdm(range(len(ds)))
    for idx in pbar:
        img_fname = ds.images[idx]
        ann_fname = ds.annotations[idx]

        # try to get image and annotation
        try:
            if try_main_fn:
                ds.__getitem__(idx) # main function for loading data
            else:
                # call specific functions from the __getitem__() method for closer investigation 
                image = Image.open(os.path.join(ds.img_dir, img_fname))
                image = image.convert('RGB')
                segmentation_map = Image.open(os.path.join(ds.ann_dir, ann_fname))

        except Exception as e:
            errmsg = f"idx {idx}:{ds.img_dir}/{img_fname}: {e}"
            print(errmsg)

            if log_errors:
                with open('errors.txt', 'a') as f:
                    f.write(f"{errmsg}\n")
            continue

def clean_dataset(errorfile, dry=True):
    imgs_to_delete = []

    with open(errorfile) as f:
        errors = f.read().split('\n')[:-1]

    for errormsg in errors:
        fname = errormsg.split(":")[1]
        imgs_to_delete.append(fname)

    assert len(imgs_to_delete) == len(errors)

    masks_to_delete = [fname.replace('.jpg', '_mask.png').replace('images/', 'annotations/') for fname in imgs_to_delete]

    for idx in range(len(imgs_to_delete)):
        img_path = imgs_to_delete[idx]
        mask_path = masks_to_delete[idx]

        print(img_path)
        print(mask_path)

        if not dry:
            os.remove(img_path)
            os.remove(mask_path)

    if dry:
        print("\nSet dry=False to remove files (--delete argument).")
