import logging
log = logging.getLogger(__name__)
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from data.ISIC import ISIC
from data.LIDC import LIDC
from data.Prostate import Prostate

class EmptyData:
    """Fake data loader for the fully-supervised case"""
    def __iter__(self):
        return self
    def __next__(self):

        return None, None


def build_dataloader(dataset, path, batch_size, transforms, shuffle = True, percSuper = None,
                     ignoreSemi = False, nSamples = 1e8):
    # Load corresponding dataset
    if dataset == 'ISIC':
        data = ISIC(path, transforms, nSamples=nSamples)
    elif dataset == 'LIDC':
        data = LIDC(path, transforms, nSamples=nSamples)
    elif dataset == 'Prostate':
        data = Prostate(path, transforms, nSamples=nSamples)
    else:
        raise NotImplementedError


    # Depending on percSuper, create a split for supervised/unsupervised data
    if percSuper is None:
        dataloader = DataLoader(data, batch_size = batch_size, shuffle = shuffle, drop_last=False)
        return dataloader
    else:
        N = len(data)
        ind = list(range(N))
        supervised = ind[:int(N*percSuper)]
        unsuper = ind[int(N*percSuper):]
        superSampler = SubsetRandomSampler(supervised)
        superLoader = DataLoader(data, batch_size=batch_size, sampler=superSampler, drop_last=True)

        if ignoreSemi:
            unsuper = []
        log.info(f'Labeled/Unlabelled Split: {len(supervised)}/{len(unsuper)}')

        # If we run fully supervised, return a fake dataloader for the semi supervised data
        if percSuper == 1 or ignoreSemi:
            return superLoader, EmptyData()

        semiSempler = SubsetRandomSampler(unsuper)
        semiLoader = DataLoader(data, batch_size=batch_size, sampler=semiSempler, drop_last=True)
        return superLoader, semiLoader
