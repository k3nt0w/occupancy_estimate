import numpy as np
from PIL import Image

def load_data(path, crop=True, size=None, mode="label", xp=np):
    img = Image.open(path)
    if crop:
        w,h = img.size
        if w < h:
            if w < size:
                img = img.resize((size, size*h//w))
                w, h = img.size
        else:
            if h < size:
                img = img.resize((size*w//h, size))
                w, h = img.size
        img = img.crop((int((w-size)*0.5), int((h-size)*0.5), int((w+size)*0.5), int((h+size)*0.5)))

    if mode=="label":
        y = xp.asarray(img, dtype=xp.int32)
        mask = y == 255
        mask = mask.astype(xp.int32)
        y[mask] = 0
        y = calc_occupancy(y)
        return y

    elif mode=="data":
        x = xp.asarray(img, dtype=xp.float32).transpose(2, 0, 1)
        x -= 120
        return x

    elif mode=="predict":
        return img

def calc_occupancy(x, n_class=21, xp=np):
    v = xp.empty((21))
    h, w = x.shape

    for i in range(n_class):
        v[i] = np.sum(x == i)
    v /= h*w
    if v.sum != 1.0:
        diff = 1.0 - v.sum()
        v[0] += diff
    return v
