import os
import time
import numpy as np
from skimage import io
import time

import glob

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from data_loader_cache import get_im_gt_name_dict, create_dataloaders, GOSRandomHFlip, GOSResize, GOSRandomCrop, GOSNormalize #GOSDatasetCache,
from models import *
import sys


from PIL import Image, ImageEnhance, ImageOps
root = os.path.dirname(os.path.realpath(__file__))




class Saliency_ISNET_Node:
    # `device` is a hint
    def __init__(self, device="cuda:0", model_name='isnet'):

        model_dir = os.path.join(root, 'saved_models', model_name, f'{model_name}.pth')
        print("...load ISNET---168 MB")
        net = ISNetDIS() 


        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_dir))
            net.cuda()
            self.device = "cuda:0"

        else:
            self.device = "cpu"

        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
        net.eval()
        self.net = net


    def __call__(self, im):
        with torch.no_grad():

            if len(im.shape) < 3:
                im = im[:, :, np.newaxis]
            if im.shape[2] == 1:
                im = np.repeat(im, 3, axis=2)
            im_tensor = torch.tensor(im, dtype=torch.float32)
            img = im_tensor.clone()
            w, h = img.shape[0], img.shape[1]

            print(im_tensor.shape)
            im_tensor = torch.transpose(torch.transpose(im_tensor,1,2),0,1)

            im_tensor = torch.unsqueeze(im_tensor,0)
            im_tensor = F.upsample(im_tensor, size=[1024,1024], mode="bilinear")

            im_tensor = im_tensor.type(torch.uint8) / 255.0


            print(im_tensor.shape)
            ds_val = self.net(im_tensor)[0]


            im_pred = F.interpolate(ds_val[0], size=(w,h))

            im_pred = torch.squeeze(im_pred)
            ma = torch.max(im_pred)
            mi = torch.min(im_pred)
            im_pred = (im_pred-mi)/(ma-mi)
            im_result = im_pred.to('cpu').detach().numpy().copy()
            # Debug code
            # pdb.set_trace()
            # im = Image.fromarray(pred)
            # im.convert('RGB').save("test.png")
            img = img.to('cpu').detach().numpy().copy()
            mask = im_result
            mask_inverse = 1 - im_result
            foreground = np.expand_dims(mask, 2) * img
            background = np.expand_dims(mask_inverse, 2) * img
            return (mask*255).astype('int'), (mask_inverse*255).astype('int'), foreground.astype('int'), background.astype('int'), mask

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    model = Saliency_ISNET_Node()

    if os.path.isdir(sys.argv[1]):
        directory = sys.argv[1]
        for filename in os.listdir(directory):
            im_path = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(im_path) and "result" not in im_path:
                img = io.imread(im_path)
                res = model(img)
                io.imsave(f"{im_path[:-4]}_result_mask_isnet_v2.png",res[0])
                io.imsave(f"{im_path[:-4]}_result_foreground_isnet_v2.png",res[2])
    else:
        for im_path in sys.argv[1:]:
            plt.clf()
            img = io.imread(im_path)
            res = model(img)

            print("max ", res[4].max())
            print("min", res[4].min())
            plt.subplot(321); plt.imshow(img)
            plt.subplot(323); plt.imshow(res[0])
            plt.subplot(324); plt.imshow(res[1])
            plt.subplot(325); plt.imshow(res[2])
            plt.subplot(326); plt.imshow(res[3])
            io.imsave(f"{im_path[:-4]}_result_mask_isnet_v2.png",res[0])
            io.imsave(f"{im_path[:-4]}_result_foreground_isnet_v2.png",res[2])
            plt.title("saliency  (RGB)")
            plt.tight_layout()
            plt.savefig("output.png", bbox_inches="tight", dpi=250)
