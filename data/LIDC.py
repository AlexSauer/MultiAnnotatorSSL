import os
import logging
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, InterpolationMode
from skimage.io import imread
log = logging.getLogger(__name__)


def normalise01(img):
    cmin = img.min()
    return (img - cmin) / (img.max() - cmin)


class LIDC(Dataset):
    def __init__(self, path, transform=None, nSamples=None):
        self.transform = transform
        self.images = []
        self.masks = []

        # Check if data can be loaded as one to make it faster
        if os.path.isfile(os.path.join(path, 'preload', 'img.pt')) and \
                os.path.isfile(os.path.join(path, 'preload', 'mask.pt')):
            log.info(f"Found preloaded files in {path}")
            images = torch.load(os.path.join(path, 'preload', 'img.pt'))
            self.images = [i[0] for i in torch.split(images, 1)]
            masks = torch.load(os.path.join(path, 'preload', 'mask.pt'))
            self.masks = [i[0] for i in torch.split(masks, 1)]

            if nSamples is not None and len(self.images) > nSamples:
                self.images = self.images[:nSamples]
                self.masks = self.masks[:nSamples]

                # Otherwise read in the data:
        else:
            # Need to resize the images from 180x180 to 160x160 to pass them through the network

            img_resize = Resize(size=(160, 160), interpolation=InterpolationMode.BILINEAR)
            mask_resize = Resize(size=(160, 160), interpolation=InterpolationMode.NEAREST)

            # Go through each patient
            patients = [d for d in os.listdir(os.path.join(path, 'images')) if not d.startswith('.')]

            for patient in tqdm(patients):
                patient_imgs = [f[:f.rfind('.')] for f in os.listdir(os.path.join(path, 'images', patient)) if
                                not f.startswith('.')]

                # Go through all images, normalise and save them and collect corresponding masks
                for img in patient_imgs:
                    d = imread(os.path.join(path, 'images', patient, img + '.png'))
                    d = torch.Tensor(normalise01(d)[None])  # [1, H, W]
                    assert d.shape == (1, 180, 180), f'Img Shape of {img} from {patient} is {d.shape}'
                    # Resize
                    d = img_resize(d)
                    assert d.shape == (1, 160, 160), f'Img Shape of {img} from {patient} is {d.shape} after resizing'
                    self.images.append(d)

                    # Save corresponding ground truth
                    cur_gt = []
                    for ending in ['_l0', '_l1', '_l2', '_l3']:
                        d = imread(os.path.join(path, 'gt', patient, img + ending + '.png'))
                        cur_gt.append(torch.Tensor(d))

                    cur_gt = torch.stack(cur_gt)  # [C, H, W]
                    assert cur_gt.shape == (4, 180, 180), f'GT Shape of {img} from {patient} is {cur_gt.shape}'
                    # Resize mask
                    cur_gt = mask_resize(cur_gt)
                    assert cur_gt.shape == (4, 160, 160), f'GT Shape of {img} from {patient} is {cur_gt.shape} after resizing'

                    cur_max = max(cur_gt.max(), 1)  # Make largest value 1 but handle all zero case
                    self.masks.append(cur_gt/cur_max)

            # Save data as one file to speed up loading later
            torch.save(torch.stack(self.images), os.path.join(path, 'preload', 'img.pt'))
            torch.save(torch.stack(self.masks), os.path.join(path, 'preload', 'mask.pt'))

        self.consistency_check()
        log.info(f'Loaded {len(self.images)} images and {len(self.masks)} masks!')

    def __len__(self):
        return len(self.images)

    def consistency_check(self):
        """Check that all the images and masks are in range [0,1]"""
        assert all([x.max() <= 1 for x in self.images]), "Some images have values larger than 1"
        assert all([x.min() >= 0 for x in self.images]), "Some images have values smaller than 0"
        assert all([x.max() <= 1 for x in self.masks]), "Some masks have values larger than 1"
        assert all([x.min() >= 0 for x in self.masks]), "Some masks have values smaller than 0"


    def __getitem__(self, k):
        if self.transform is None:
            return self.images[k], self.masks[k]
        else:
            return self.transform(self.images[k], self.masks[k])

