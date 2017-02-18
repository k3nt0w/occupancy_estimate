from chainer import serializers
import numpy as np
from PIL import Image
import os
import argparse
import cv2

from model import VGG
from preprocess import load_data


parser = argparse.ArgumentParser(description='Chainer Fully Convolutional Network: predict')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--image_path', '-i', default=None, type=str)
parser.add_argument('--weight', '-w', default="weight/chainer_fcn.weight", type=str)
parser.add_argument('--classes', default=21, type=int)
parser.add_argument('--clop', "-c", default=True, type=bool)
parser.add_argument('--clopsize', "-s", default=224, type=int)
args = parser.parse_args()

img_name = args.image_path.split("/")[-1].split(".")[0]

model = VGG(n_class=args.classes)
serializers.load_npz('weight/weight', model)

x = load_data(args.image_path, crop=args.clop, size=args.clopsize, mode="data")
x = np.expand_dims(x, axis=0)
pred = model(x).data
print(pred[0])
