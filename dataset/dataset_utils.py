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