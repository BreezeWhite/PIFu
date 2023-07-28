import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import torch

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.model import *

from PIL import Image
import torchvision.transforms as transforms
import glob
import tqdm

# get options
opt = BaseOptions().parse()

class Evaluator:
    def __init__(self, opt, projection_mode='orthogonal'):
        self.opt = opt
        self.load_size = self.opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # set cuda
        cuda = torch.device('cuda:%d' % opt.gpu_id) if torch.cuda.is_available() else torch.device('cpu')

        # create net
        netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
        print('Using Network: ', netG.name)

        if opt.load_netG_checkpoint_path:
            netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

        if opt.load_netC_checkpoint_path is not None:
            print('loading for net C ...', opt.load_netC_checkpoint_path)
            netC = ResBlkPIFuNet(opt).to(device=cuda)
            netC.load_state_dict(torch.load(opt.load_netC_checkpoint_path, map_location=cuda))
        else:
            netC = None

        os.makedirs(opt.results_path, exist_ok=True)
        os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

        opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
        with open(opt_log, 'w') as outfile:
            outfile.write(json.dumps(vars(opt), indent=2))

        self.cuda = cuda
        self.netG = netG
        self.netC = netC

    def crop_image(self, image_arr, mask_arr):
        idy, idx = np.where(mask_arr > 30)
        min_y, max_y, min_x, max_x = np.min(idy), np.max(idy), np.min(idx), np.max(idx)
        sub_img = image_arr[min_y:max_y, min_x:max_x]
        sub_mask = mask_arr[min_y:max_y, min_x:max_x]

        diff_y = max_y - min_y
        diff_x = max_x - min_x
        diff = max(diff_y, diff_x)
        radius = int(diff * 1.1 // 2)

        new_img = np.zeros((radius * 2, radius * 2, 3), dtype=np.uint8)
        new_mask = np.zeros((radius * 2, radius * 2), dtype=np.uint8)

        y_center = (min_y + max_y) // 2
        x_center = (min_x + max_x) // 2
        paste_start_y = radius - (y_center - min_y)
        paste_start_x = radius - (x_center - min_x)

        slice_idx = np.s_[paste_start_y:paste_start_y + diff_y, paste_start_x:paste_start_x + diff_x]
        new_img[slice_idx] = sub_img
        new_mask[slice_idx] = sub_mask

        return new_img, new_mask

    def load_image(self, image_path, mask_path):
        # Name
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        # Calib
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib = torch.Tensor(projection_matrix).float()

        # Load image & mask
        mask = np.array(Image.open(mask_path).convert('L'))
        image = np.array(Image.open(image_path).convert('RGB'))

        # Crop image & mask
        image, mask = self.crop_image(image, mask)

        # Convert numpy array back to Image
        mask = Image.fromarray(mask)
        image = Image.fromarray(image)

        # Transform
        mask = transforms.Resize(self.load_size)(mask)
        mask = transforms.ToTensor()(mask).float()
        image = self.to_tensor(image)
        image = mask.expand_as(image) * image
        return {
            'name': img_name,
            'img': image.unsqueeze(0),
            'calib': calib.unsqueeze(0),
            'mask': mask.unsqueeze(0),
            'b_min': B_MIN,
            'b_max': B_MAX,
        }

    def eval(self, data, use_octree=False):
        '''
        Evaluate a data point
        :param data: a dict containing at least ['name'], ['image'], ['calib'], ['b_min'] and ['b_max'] tensors.
        :return:
        '''
        opt = self.opt
        with torch.no_grad():
            self.netG.eval()
            if self.netC:
                self.netC.eval()
            save_path = '%s/%s/result_%s.obj' % (opt.results_path, opt.name, data['name'])
            if self.netC:
                gen_mesh_color(opt, self.netG, self.netC, self.cuda, data, save_path, use_octree=use_octree)
            else:
                gen_mesh(opt, self.netG, self.cuda, data, save_path, use_octree=use_octree)


if __name__ == '__main__':
    evaluator = Evaluator(opt)

    img_folder = Path(opt.test_folder_path)
    test_images = [
        f
        for f in img_folder.glob('*')
        if f.suffix.lower() in ('.jpg', 'jpeg', '.png') and (not 'mask' in str(f))
    ]
    test_masks = [str(f)[:-4]+'_mask.png' for f in test_images]

    print("num: ", len(test_masks))

    for image_path, mask_path in tqdm.tqdm(zip(test_images, test_masks)):
        print(image_path, mask_path)
        data = evaluator.load_image(image_path, mask_path)
        evaluator.eval(data, True)
