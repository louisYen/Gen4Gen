import os
from pathlib import Path
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
import sys
import numpy as np
from PIL import Image
import glob


from saliency_models.U2Net.data_loader import RescaleT
from saliency_models.U2Net.data_loader import ToTensor
from saliency_models.U2Net.data_loader import ToTensorLab
from saliency_models.U2Net.data_loader import SalObjDataset

from saliency_models.U2Net.model import U2NET # full size version 173.6 MB
from saliency_models.U2Net.model import U2NETP # small version u2net 4.7 MB
import pdb

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

from PIL import Image, ImageEnhance, ImageOps

root = os.path.dirname(os.path.realpath(__file__))


def rescale_output(w,h,pred):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    predict_np_im = Image.fromarray(predict_np)
    imo = predict_np_im.resize((w,h),resample=Image.BILINEAR)
    return np.array(imo)



class Saliency_U2Net_Node:
    # `device` is a hint
    def __init__(self, model_path: str, device: str="cuda:0", model_name: str='u2net'):

        model_dir = os.path.join(model_path, 'saved_models', model_name, f'{model_name}.pth')

        if(model_name=='u2net'):
            print("...load U2NET---173.6 MB")
            net = U2NET(3,1)
        elif(model_name=='u2netp'):
            print("...load U2NEP---4.7 MB")
            net = U2NETP(3,1)

        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_dir))
            net.cuda()
            self.device = "cuda:0"

        else:
            self.device = "cpu"

        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
        net.eval()
        self.net = net
        self.img_transform = transforms.Compose([RescaleT(320),
                                        ToTensorLab(flag=0)])


    def __call__(self, img):
        with torch.no_grad():
            if isinstance(img, np.ndarray):
                if img.shape[2] == 1:
                    img = Image.fromarray(img, mode="L")
                elif img.shape[2] == 3:
                    img = Image.fromarray(img, mode="RGB")
                elif img.shape[2] == 4:
                    img = Image.fromarray(img, mode="RGBA")

            img = img.convert('RGB')
            img = np.array(img)
            w, h = img.shape[0], img.shape[1]

            sample = {'imidx':np.array([0]), 'image':img, 'label':np.zeros(img.shape)}
            sample = self.img_transform(sample)
            
            img_var=sample["image"].unsqueeze(0).to(self.device).type(torch.FloatTensor)
            d1,d2,d3,d4,d5,d6,d7 = self.net(img_var)
            pred = d1[:,0,:,:]
            pred = normPRED(pred)
            rescaled_pred = rescale_output(h,w,pred)
            im_result =  rescaled_pred

            mask = im_result
            mask_inverse = 1 - mask
            foreground = np.expand_dims(mask, 2) * img
            background = np.expand_dims(mask_inverse, 2) * img
            # print(mask.shape)
            # print(foreground.shape)
            # print(np.max(mask))
            # print(np.min(mask))
            # print(np.max(img))
            return (mask*255).astype('int'),   mask #(mask*255).astype('int')*0, background.astype('int')[:,:,0]*0, (mask*255).astype('int')*0, foreground.astype('int')[:,:,0]


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    model = Saliency_U2Net_Node()

    if os.path.isdir(sys.argv[1]):
        directory = sys.argv[1]
        for filename in os.listdir(directory):
            im_path = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(im_path) and "result" not in im_path:
                img = io.imread(im_path)
                res = model(img)
                io.imsave(f"{im_path[:-4]}_result_mask_u2net.png",res[0])
                io.imsave(f"{im_path[:-4]}_result_foreground_u2net.png",res[2])
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
            io.imsave(f"{im_path[:-4]}_result_mask_u2net.png",res[0])
            io.imsave(f"{im_path[:-4]}_result_foreground_u2net.png",res[2])
            plt.title("saliency  (RGB)")
            plt.tight_layout()
            plt.savefig("output.png", bbox_inches="tight", dpi=250)
