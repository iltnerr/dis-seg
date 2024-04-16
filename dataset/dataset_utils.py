import numpy as np
from PIL.Image import fromarray

cls_dict = {
    0: 'Background',
    1: 'Building',
    2: 'Road',
    3: 'Vehicle',
    4: 'Debris',
    5: 'Fire',
    6: 'Water',
    7: 'Animal',
    8: 'Injured_Person', # Keeping this class for convenience, since it is part of the annotations. With a small dataset it does not make sense to distinguish healthy and injured persons, however.
    9: 'Sky',
    10: 'Smoke',
    11: 'Tree',
    12: 'Person'
}

def pixel_values_to_pil_image(pixel_values):
    np_arr = np.array(pixel_values)
    reordered = np.moveaxis(np_arr, 0, -1)
    rescaled = ((reordered - reordered.min()) * (1/(reordered.max() - reordered.min()) * 255)).astype('uint8')
    return fromarray(rescaled)
        



