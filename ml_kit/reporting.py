import os
import re
from pathlib import Path
import numpy as np
import imageio


def pick_random_pairs(paired_arrays, amount):
    result = []
    if len(paired_arrays) == 0:
        return result
    if hasattr(paired_arrays, 'shape'):
        is_numpy = True
        length = paired_arrays[0].shape[0]
    else:
        is_numpy = False
        length = len(paired_arrays[0])
    if is_numpy:
        if any(v.shape[0] != length for v in paired_arrays[1:]):
            raise ValueError("Paired arrays should match to each other on first dimension!")
    else:
        if any(len(v) != length for v in paired_arrays[1:]):
            raise ValueError("Paired arrays should match to each other on first dimension!")
    if length == 0:
        return result
    for i in range(amount):
        idx = np.random.randint(0, length)
        for arr in paired_arrays:
            result.append(arr[idx])
    return result

    # NOTE: use case:
    # imgs = [img.squeeze() for img pick_random_pairs([x_test, y_pred], 5)]
    # plt.figure(figsize=(14, 7))
    # plot_images(imgs, 2, 5, ordering=S_COLS)
    # plt.show()


def animate_imgs(input_path, filename_regex, output_path, recurse=False):
    images = []
    files_list = []
    path = Path(input_path)
    for root,_,files in os.walk(path):
        for fn in files:
            if re.match(filename_regex, fn):
                files_list.append(root / fn)
        if not recurse:
            break
    files_list.sort()
    for fp in files_list:
        images.append(imageio.imread(fp))
    imageio.mimsave(output_path, images)

    # NOTE: use case:
    # from IPython.display import Image
    # animate_imgs(some_path, "img_\d\d\d\d\.jpg", some_path / 'anim.gif')
    # Image(open(some_path / 'anim.gif','rb').read())
