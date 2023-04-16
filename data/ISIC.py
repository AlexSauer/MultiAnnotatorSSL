import logging
log = logging.getLogger(__name__)
import numpy as np
import tifffile
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class ISIC(Dataset):
    def __init__(self, path, transform = None, nSamples = 1e8):
        """
        Builds and returns the dataloader
        :param path: path to a directory that holds the data with subdirectories img/ and mask/
        Each file is assumed to be a tif file and we assume that img are RGB images and mask are binary images but have a
        separate channel for each annotation available. Images and masks have the same name
        :param transform: Class which implements a __call__ method that applies some data augmentation to an image
        and mask at the same time
        :return:
        Main method is __getitem__ which returns two tensors which shape [C, H, W] after applying the given
        transformation
        """

        self.transform = transform

        # Check if data can be loaded as one to make it faster
        if os.path.isfile(os.path.join(path, 'preload', 'img.pt')) and \
                os.path.isfile(os.path.join(path, 'preload', 'mask.pt')):
            log.info(f"Found preloaded files in {path}!")
            images = torch.load(os.path.join(path, 'preload', 'img.pt'))
            self.images = [i[0] for i in torch.split(images, 1)]
            masks = torch.load(os.path.join(path, 'preload', 'mask.pt'))
            masks = masks / masks.max()
            self.masks = [i[0] for i in torch.split(masks, 1)]

            if len(self.images) > nSamples:
                self.images = self.images[:nSamples]
                self.masks = self.masks[:nSamples]

        else:
            # Build dict of all images
            img_files = [f for f in os.listdir(os.path.join(path, 'img')) if f.endswith('.tif') and not f.startswith('.')]
            mask_files = [f for f in os.listdir(os.path.join(path, 'mask')) if f.endswith('.tif') and not f.startswith('.')]
            assert set(img_files) == set(mask_files), f'Image files and mask files do not match!'

            # Sort by names to make reproducible
            img_files.sort()

            self.images = []
            self.masks = []

            logging.info("Reading in files...")
            for i, file in tqdm(enumerate(img_files)):
                if i > nSamples:
                    break
                img = tifffile.imread(os.path.join(path, 'img', file))
                img = np.transpose(img, (2,0,1))  # Format (channel, W, H)
                img = torch.Tensor(img)

                mask = tifffile.imread(os.path.join(path, 'mask', file))
                # If we have multiple annotators, they are given by different channels and we reange to [C, W, H]
                if len(mask.shape) == 3:
                    mask = np.transpose(mask, (2,0,1))  # Format (channel, W, H)

                    # We need the same number of annotators per file (for dataloader to work)
                    # Ignore files with only 1 (problem anyway) and for the ones with more than two,
                    # just pick the first 2 (1015 files with 2, 71 with 3, 31 with 4, and 3 with 5 annotators)
                    if mask.shape[0] == 1:
                        continue
                    if mask.shape[0] > 2:
                        mask = mask[:2]
                elif len(mask.shape) == 2:
                    mask = mask[None]
                else:
                    raise NotImplementedError
                mask = torch.Tensor(mask)


                self.masks.append(mask)
                self.images.append(img)

            # Save data as one file to speed up loading later
            torch.save(torch.stack(self.images), os.path.join(path, 'preload', 'img.pt'))
            torch.save(torch.stack(self.masks), os.path.join(path, 'preload', 'mask.pt'))

        # Check that the range is correct
        self.consistency_check()
        log.info(f'Loaded {len(self.images)} images and {len(self.masks)} masks!')



    def __len__(self):
        return len(self.images)

    def __getitem__(self, k):
        if self.transform is None:
            return self.images[k], self.masks[k]
        else:
            return self.transform(self.images[k], self.masks[k])

    def consistency_check(self):
        """Check that all the images and masks are in range [0,1]"""
        assert all([x.max() <= 1 for x in self.images]), "Some images have values larger than 1"
        assert all([x.min() >= 0 for x in self.images]), "Some images have values smaller than 0"
        assert all([x.max() <= 1 for x in self.masks]), "Some masks have values larger than 1"
        assert all([x.min() >= 0 for x in self.masks]), "Some masks have values smaller than 0"


if __name__ == '__main__':
    # Small test
    from Transforms import Augmenter
    path = '/well/rittscher/users/jyo949/data/ISIC2018/mult_Anno/'
    transforms = Augmenter(['rotate', 'flip', 'gaussian_noise'])

    data = ISIC(path, transforms)
    print(len(data))

    import matplotlib.pyplot as plt
    img, mask = data[0]
    plt.imshow(img.permute((1,2,0)))
    # plt.imshow(mask[0])
    plt.show()
    print('Finished!')


