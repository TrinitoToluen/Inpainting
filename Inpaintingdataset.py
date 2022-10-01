import sys
import torch.utils.data as data
from os import listdir
from Inpaintingtools import default_loader, is_image_file, normalize
import os
import torchvision.transforms as transforms

class Dataset(data.Dataset):
    def __init__(self, data_path, image_shape, with_subfolder=False, random_crop=True, return_name=False):
        super(Dataset, self).__init__()
        if with_subfolder:
            self.samples = self._find_samples_in_subfolders(data_path)
            print("with sub_folder")
        else:
            self.samples = [x for x in listdir(data_path) if is_image_file(x)]                
        self.data_path = data_path
        self.image_shape = image_shape[:-1]
        self.random_crop = random_crop
        self.return_name = return_name

    def __getitem__(self, index):
        path = os.path.join(self.data_path, self.samples[index])
        img = default_loader(path)
        if self.random_crop:
            imgw, imgh = img.size
            if imgh < self.image_shape[0] or imgw < self.image_shape[1]:
                img = transforms.Resize(min(self.image_shape))(img)
            img = transforms.RandomCrop(self.image_shape)(img)
        else:
            img = transforms.Resize(self.image_shape)(img)
            img = transforms.RandomCrop(self.image_shape)(img)
        img = transforms.ToTensor()(img) 
        img = normalize(img)
        if self.return_name:
            return self.samples[index], img
        else:
            return img

    def _find_samples_in_subfolders(self, dir):
        if sys.version_info >= (3, 5): 
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
            print("d")
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        samples = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        samples.append(path)
        return samples

    def __len__(self):
        return len(self.samples)
       
dataset = Dataset(data_path =r'C:\Users\MHC\Documents\Python\Dataset\img_align_celeba\img_align_celeba',
                                with_subfolder=False,
                                image_shape=[218, 178, 3],
                                random_crop=True)
