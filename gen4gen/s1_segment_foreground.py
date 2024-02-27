import os
import sys
import shutil
import imghdr
import argparse
import numpy as np
import os.path as osp

from pathlib import Path
from typing import Optional
from skimage import io, transform
from matplotlib import pyplot as plt

from rich.console import Console
from rich.table import Column
from rich.progress import track, Progress, BarColumn, TextColumn

from saliency_models.DIS import Saliency_ISNET_Node
from saliency_models.U2Net import Saliency_U2Net_Node

VALID_IMAGE_EXTENSION = (
    'rgb', 'gif', 'pbm', 'pgm', 'ppm', 'tiff', 'rast',
    'xbm', 'jpeg', 'jpg', 'bmp', 'png', 'webp', 'ext')

def parse_args():
    parser = argparse.ArgumentParser(description='Saliency Object Detection')
    parser.add_argument('--src-dir', default='', help='source directory')
    parser.add_argument('--src-imgs', default='', nargs='+', help='source images')
    parser.add_argument('--tmp-dir', default='../data/s1_segmented_raw', help='temporarily root')
    parser.add_argument('--dest', default='', help='dest root')
    parser.add_argument('--rounds', type=int, default=3, help='how many rounds to extract foreground')
    parser.add_argument('--tar-ext', type=str, default='png', help='saved format (default: png)')
    parser.add_argument('--isnet-model-name', type=str, default='isnet-general-use',
            help='checkpoint name')
    args = parser.parse_args()
    return args

class SaliencyNode:
    def __init__(self,
                 model_path: str = osp.join('saliency_models', 'DIS'), # model path
                 device: str = "cuda:0",
                 model_name: str = "isnet"):
        self.device = device
        self.models = {}
        self.init(model_path, model_name)

    def init(self, model_path: str='saliency_models', model_name: str='u2net'):
        if model_name.startswith("u2net"):
            self.models[model_name] = Saliency_U2Net_Node(model_path, device=self.device, model_name=model_name)
        elif model_name.startswith("isnet"):
            self.models[model_name] = Saliency_ISNET_Node(model_path, device=self.device, model_name=model_name)
        else:
            print(f"[SaliencyNode] ERROR Unrecognized model name. Choose from [u2net|isnet]. You provided {model_name}")

    def __call__(self, img, model_name='isnet-generatl-use'):
        if model_name in self.models.keys():
            return self.models[model_name](img)
        else:
            self.init(model_name)
            return self.models[model_name](img)


def run_image(filename, model,
              dest: Optional[str] = None, # target directory
              n_round: int = 0, # number of round for segmenting foreground
              ext: str = 'png', # output file extension
              isnet_model_name: str = 'isnet-general-use', # model name
              outname_tmpl: dict = dict(
                                    fg='result_foreground_isnet', # output foreground filename template
                                    mask='result_mask_isnet') # output mask filename template
    ):

    # Check if it is a file
    if osp.isfile(filename):# and "result" not in filename:
        img_name = Path(filename).stem

        # Segment foreground
        img = io.imread(filename)

        if img.ndim == 2: # gray-scale image
            img = np.stack([img]*3, axis=-1)

        res = model(img, model_name=isnet_model_name)

        if outname_tmpl['fg'] in img_name:
            img_name = img_name.replace('_' + outname_tmpl['fg'], '')

        # Save mask
        io.imsave(osp.join(dest, f"{img_name}_{outname_tmpl['mask']}.{ext}"), res[0].astype(np.uint8))

        saliency_mask = res[0].astype(np.float32) / 255. # turn to 0-1 values
        segmented_fg = np.expand_dims(saliency_mask, 2) * img.astype(np.float32) # mask out background
        # Save segmented foreground
        io.imsave(osp.join(dest, f"{img_name}_{outname_tmpl['fg']}.{ext}"),
                    segmented_fg.astype(np.uint8))

if __name__ == '__main__':

    args = parse_args()

    # Initialize the Saliency Detector (default: isnet-general-use)
    model = SaliencyNode(model_path=osp.join('saliency_models', 'DIS'), model_name=args.isnet_model_name)

    # Create directory if not exists
    if bool(args.dest):
        Path(args.dest).mkdir(exist_ok=True, parents=True)

    category = None
    outname_tmpl: dict = dict(fg='result_foreground_isnet', mask='result_mask_isnet')

    console = Console()
    console.rule("[bold blue]1️⃣  Step1. Object Association and Foreground Segmentation")

    if osp.isdir(args.src_dir):

        all_files = list()
        for path, directories, files in os.walk(args.src_dir, topdown=True):

            # Only consider files with valid image extension
            files = sorted([f for f in files if f.lower().endswith(VALID_IMAGE_EXTENSION)])

            # Ignore non-image files

            if not bool(files): continue

            # Collect files with valid image extension
            for filename in files:
                img_path = osp.join(path, filename)
                all_files.append((path, img_path))

        for (path, img_path) in track(all_files,
                                      description="[green]Segment source images from the directory..."):

            # Get folder name from each scene
            scene_folder_name = Path(path).parents[0].stem

            category = Path(path).stem

            for n in range(args.rounds):
                subdir = f'round_{n}'
                tmp_dir = osp.join(args.tmp_dir, scene_folder_name, subdir, category)
                Path(tmp_dir).mkdir(exist_ok=True, parents=True)

                if n == 0:
                    # At round 0: use original source images including background
                    run_image(img_path,
                              model,
                              dest=tmp_dir,
                              ext=args.tar_ext,
                              outname_tmpl=outname_tmpl,
                              isnet_model_name=args.isnet_model_name)
                else:

                    # Use segmented image as input from previous round
                    filename = Path(img_path).stem.split('.')[0]
                    src_dir = tmp_dir.replace(f'round_{n}', f'round_{n-1}') # load image from previous round
                    src_path = osp.join(src_dir, f"{filename}_{outname_tmpl['fg']}.{args.tar_ext}")

                    run_image(src_path,
                              model,
                              dest=tmp_dir,
                              ext=args.tar_ext,
                              outname_tmpl=outname_tmpl,
                              isnet_model_name=args.isnet_model_name)

                    if n == args.rounds - 1: # the last round
                        out_dest = osp.join(args.dest, scene_folder_name, category)
                        shutil.copytree(tmp_dir, out_dest, dirs_exist_ok=True)

    else:
        category = None
        dest = args.dest
        for im_path in track(args.src_imgs, description="[blue]Segment images..."):
            run_image(im_path, model, dest=dest, isnet_model_name=args.isnet_model_name)
