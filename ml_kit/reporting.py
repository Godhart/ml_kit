import os
import re
from pathlib import Path
import numpy as np
import imageio

from IPython.display import display, Markdown, Latex

import sys
from pathlib import Path

ml_kit_path = str((Path(__file__).absolute() / ".." / "..").resolve())
if ml_kit_path not in sys.path:
    sys.path.insert(0, ml_kit_path)

try:
    from ml_kit.standalone import STANDALONE
except ImportError:
    STANDALONE = False

if STANDALONE:
    from ml_kit.env import *

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
    # imgs = [img.squeeze() for img in pick_random_pairs([x_test, y_pred], 5)]
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


# NOTE: printing to markdown tips
# display(Markdown('*some markdown* $\phi$'))
# # If you particularly want to display maths, this is more direct:
# display(Latex('\phi'))


def print_markdown(message):
    if ENV[ENV__JUPYTER]:
        display(Markdown(message))
    else:
        print(message)


def print_latex(expr):
    if ENV[ENV__JUPYTER]:
        display(Latex(expr))
    else:
        print(expr)


def print_table(header, data):
    message = []
    message.append("")
    message.append("| " + " | ".join(str(v) for v in header) + " |")
    message.append("|-" + "-|-".join("-" for v in header) + "-|")
    for item in data:
        message.append("| " + " | ".join(str(v) for v in item) + " |")
    message.append("")
    message = "\n".join(message)
    print_markdown(message)


def dict_to_table(data, header=None, dict_key=None, default='', sort_key=None):
    if dict_key is None:
        dict_key = '_key_'
    if header is None:
        header = [dict_key]
        for k, v in data.items():
            for kk in v:
                if kk not in header:
                    header.append(kk)

    rows = []
    sorted_keys = sorted(data.keys(), key=sort_key)
    for k  in sorted_keys:
        row = [default]*len(header)
        if dict_key in header:
            row[header.index(dict_key)] = k
        for kk, vv in data[k].items():
            if kk not in header:
                continue
            row[header.index(kk)] = vv
        rows.append(row)

    return header, rows
