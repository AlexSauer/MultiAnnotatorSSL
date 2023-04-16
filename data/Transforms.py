import logging
log = logging.getLogger(__name__)
import torch

class AugComposer:
    """Composing several augmentation methdos"""
    def __init__(self, trans_list):
        self.trans = trans_list
        self.exec = {}

    def __call__(self, img, mask):
        self.exec = {}
        img_aug, mask_aug = img.clone(), mask.clone()
        for t in self.trans:
            self.exec[type(t).__name__] = t.exec
            img_aug, mask_aug = t(img_aug, mask_aug)
        return img_aug, mask_aug

class AugOneOf:
    """Picking one out of a list of given augmentations"""
    def __init__(self, trans_list):
        self.trans = trans_list

    def __call__(self, img, mask):
        ind = torch.randint(low = 0, high = len(self.trans), size=(1,)).item()
        return self.trans[ind](img, mask)


class Trans:
    """
    Base class for my augmentation. Expects two tensors (img and mask) with shape [C, W, H] and [C, W, H]
    Argument p gives the probability of applying the given transformation.
    """

    def __init__(self, p):
        self.p = p
        self.exec = False

    def __call__(self, x, mask):
        raise NotImplementedError


class Rotate90(Trans):
    """Rotates the last two dimensions"""
    def __call__(self, x, mask):
        img_d, mask_d = len(x.shape)-1, len(mask.shape)-1
        if torch.rand(1) < self.p:
            self.exec = True
            k_rot = torch.randint(low=1, high=5, size=(1,)).item()
            return torch.rot90(x, k=k_rot, dims=(img_d-1, img_d)), torch.rot90(mask, k=k_rot, dims=(mask_d-1, mask_d))
        else:
            self.exec = False
            return x, mask

class VFlip(Trans):
    """Flips the last dimensions"""
    def __call__(self, x, mask):
        img_d, mask_d = len(x.shape)-1, len(mask.shape)-1
        if torch.rand(1) < self.p:
            self.exec = True
            return  torch.flip(x, dims =(img_d,)), torch.flip(mask, dims =(mask_d,))
        else:
            self.exec = False
            return x, mask


class HFlip(Trans):
    """Flips the second to last dimension"""
    def __call__(self, x, mask):
        img_d, mask_d = len(x.shape)-1, len(mask.shape)-1
        if torch.rand(1) < self.p:
            self.exec = True
            return  torch.flip(x, dims =(img_d-1,)), torch.flip(mask, dims =(mask_d-1,))
        else:
            self.exec = False
            return x, mask


class GaussianNoise(Trans):
    def __call__(self, x, mask, sigma = 0.05, max_v = 1):
        """By default assumes that the pixel values are in [0, 1]"""
        if torch.rand(1) < self.p:
            self.exec = True
            noise = torch.randn(x.shape) * max_v * sigma
            return  x + noise, mask
        else:
            self.exec = False
            return x, mask


class GaussianNoisePartial(Trans):
    """Applies the Gaussian noise only to a random subsample of pixels"""
    def __call__(self, x, mask, sigma = 0.25, max_v = 1, partial_p = 0.1):
        if torch.rand(1) < self.p:
            self.exec = True
            pixels = (torch.rand(x.shape, device = x.device) < partial_p).float()
            noise = torch.randn(x.shape, device = x.device) * max_v * sigma
            return torch.clamp(x + pixels * noise, min = 0, max=max_v), mask
        else:
            self.exec = False
            return x, mask


class ElasticTrans(Trans):
    def __call__(self, x):
        raise NotImplementedError


class Augmenter:
    def __init__(self, transforms = [], verbose = True):
        org_trans = transforms[:]
        trans_list = []
        if 'rotate' in transforms:
            trans_list.append(Rotate90(p=0.5))
            transforms.remove('rotate')
        if 'flip' in transforms:
            trans_list.append(VFlip(p=0.5))
            trans_list.append(HFlip(p=0.5))
            transforms.remove('flip')
        if 'elastic' in transforms:
            raise NotImplementedError
            trans_list.append(ElasticTransform(alpha=20, sigma=20 * 0.16, alpha_affine=0, p=0.1))
            transforms.remove('elastic')
        if 'gaussian_noise' in transforms:
            trans_list.append(GaussianNoisePartial(p = 0.5, ))
            transforms.remove('gaussian_noise')
        if 'gaussian_blur' in transforms:
            raise NotImplementedError

        # If we supplied any transforms check that all of them have been implemented
        if org_trans:
            assert not transforms, f'Some transform cannot be implemented: {transforms}'

        if verbose:
            log.info(f'Random transformations: {org_trans}')

        self.aug = AugComposer(trans_list) if trans_list else None

    def __call__(self, img, mask = None):
        """We expect img and mask to have shape [B,C,H,W] and [B,H,W]"""
        if self.aug is None:
            if mask is None:
                return img
            else:
                return img, mask

        #assert len(img.shape) == 4, 'Image needs to have shape [B, C, H, W]'

        if mask is not None:
            assert len(mask.shape) == 3, 'Mask is expected to have shape [B, H, W]'
            return self.aug(img, mask)  # Return augmented_imge, augmented_mask

        else:
            return self.aug(img, torch.zeros(img.shape))[0]
