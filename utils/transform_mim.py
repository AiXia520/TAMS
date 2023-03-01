# coding=utf-8
from __future__ import absolute_import, division, print_function
from torchvision import transforms
from data.data_list_image import Normalize
import numpy as np

from utils.auto_augment import AutoAugment

class MaskGenerator:
    def __init__(self, input_size, mask_patch_size, model_patch_size, mask_ratio):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2

        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):

        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


class DataAugmentation:
    def __init__(self, weak_transform, strong_transform, args):
        self.transforms = [weak_transform, strong_transform]

        self.mask_generator = MaskGenerator(
            input_size= args.img_size ,
            mask_patch_size= args.mask_patch_size,
            model_patch_size= args.model_patch_size,
            mask_ratio= args.mask_ratio,
        )

    def __call__(self, x):
        images_weak = self.transforms[0](x)
        images_strong = self.transforms[1](x)

        return images_weak, images_strong, self.mask_generator()


def get_transform(dataset, img_size,args):
    if dataset in ['svhn2mnist', 'usps2mnist', 'mnist2usps']:
        transform_source = transforms.Compose([
                transforms.RandomResizedCrop((img_size, img_size), scale=(0.75, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        transform_target = transforms.Compose([
                transforms.RandomResizedCrop((img_size, img_size), scale=(0.75, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        transform_test = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
    elif dataset in ['visda17', 'office-home']:
        transform_source_weak = transforms.Compose([
                transforms.Resize((img_size+32, img_size+32)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Normalize(meanfile='./data/ilsvrc_2012_mean.npy')
        ])

        # transform_list = [
        #     transforms.Resize((img_size + 32, img_size + 32)),
        #     transforms.RandomCrop(img_size),
        #     transforms.RandomHorizontalFlip(),
        # ]
        # transform_list.append(AutoAugment())
        # transform_list.extend([transforms.ToTensor(), Normalize(meanfile='./data/ilsvrc_2012_mean.npy')])
        # transform_source_strong = transforms.Compose(transform_list)


        transform_source_strong = transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Normalize(meanfile='./data/ilsvrc_2012_mean.npy')
        ])

        transform_test = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                Normalize(meanfile='./data/ilsvrc_2012_mean.npy')
            ])
        transform_source_mim = DataAugmentation(transform_source_weak, transform_source_strong, args)

    else:
        transform_source_weak = transforms.Compose([
                transforms.Resize((img_size+32, img_size+32)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Normalize(meanfile='./data/ilsvrc_2012_mean.npy')
        ])
        transform_source_strong = transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Normalize(meanfile='./data/ilsvrc_2012_mean.npy')
        ])

        transform_source_mim = DataAugmentation(transform_source_weak, transform_source_strong, args)

        transform_test = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                Normalize(meanfile='./data/ilsvrc_2012_mean.npy')
            ])

    return transform_source_mim, transform_source_mim, transform_test





