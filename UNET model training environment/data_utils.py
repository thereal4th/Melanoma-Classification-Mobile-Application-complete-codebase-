import os
import sys
import random
import numpy as np
from PIL import Image
from pydensecrf import densecrf
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

def dense_crf(inputs, predict_probs):
    h = predict_probs.shape[0]
    w = predict_probs.shape[1]
    
    predict_probs = np.expand_dims(predict_probs, 0)
    predict_probs = np.append(1 - predict_probs, predict_probs, axis=0)
    
    d = densecrf.DenseCRF2D(w, h, 2)
    U = -np.log(predict_probs)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    inputs = np.ascontiguousarray(inputs)
    
    d.setUnaryEnergy(U)
    
    d.addPairwiseGaussian(sxy=20, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=inputs, compat=10)
    
    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))
    
    return Q

class ImageDataset(Dataset):
    def __init__(self, fdir, bdir, imsize, mode, aug_prob=0.5, prefetch=False):
        self._fdir     = fdir
        self._mask_dir = bdir   # folder containing your *_segmentation.png masks
        self._resize   = T.Resize((300, 300))
        self._mode     = mode
        self._aug_prob = aug_prob
        self._rot_degs = [0, 90, 180, 270]
        self._impaths  = sorted(os.path.join(fdir, fn) for fn in os.listdir(fdir))
        print(f"{mode} images: {len(self._impaths)}")

    def __len__(self):
        return len(self._impaths)

    def __getitem__(self, idx):
        impath  = self._impaths[idx]
        augment = (self._mode == 'train' and random.random() < self._aug_prob)
        return self._transform_img(impath, augment)

    def _transform_img(self, impath, augment):
        # load image
        img = Image.open(impath).convert('RGB')

        # build mask path (only this convention)
        stem      = os.path.splitext(os.path.basename(impath))[0]
        mask_path = os.path.join(self._mask_dir, f"{stem}_segmentation.png")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Expected mask at {mask_path}")
        mask = Image.open(mask_path).convert('L')

        # resize both
        img, mask = self._resize(img), self._resize(mask)

        # same spatial augment on both
        if augment:
            rot = random.choice(self._rot_degs)
            img  = T.RandomRotation(rot)(img)
            mask = T.RandomRotation(rot)(mask)
            if random.random() < 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() < 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)
            # color jitter only on image
            img = T.ColorJitter(0.2,0.2,0.02)(img)

        # to-tensor + normalize / binarize
        img  = T.ToTensor()(img)
        img  = T.Normalize((0.5,)*3, (0.5,)*3)(img)
        mask = T.ToTensor()(mask)
        mask = (mask > 0.5).float()

        return img, mask

class TestImageDataset(Dataset):
    def __init__(self, fdir, imsize=300):
        self._resize = T.Resize((imsize, imsize))
        self._imsize = imsize
        self._fdir = fdir
        self._impaths = sorted([os.path.join(fdir, fname) for fname in os.listdir(fdir)])

        # Transformation for test images
        self._transform = T.Compose([
            self._resize,
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self._osize = []
        for file in self._impaths:
            img = Image.open(file).convert('RGB')
            self._osize.append(img.size)

        print(f"Image count in test path: {len(self._impaths)}")
    
    def __getitem__(self, idx):
        impath = self._impaths[idx]
        img = Image.open(impath).convert('RGB')
        img = self._transform(img)
        return idx, img

    def __len__(self):
        return len(self._impaths)
    
    def save_img(self, index, predict, use_crf):
        predict = predict.squeeze().cpu().numpy()
        if use_crf:
            inputs = self._dataset[index].permute(1, 2, 0).numpy()
            predict = dense_crf(np.array(inputs).astype(np.uint8), predict)
        predict = np.array((predict > 0.5) * 255).astype(np.uint8)
        mask = Image.fromarray(predict, mode='L')
        mask = mask.resize(self._osize[index])
        fg = Image.new('RGB', self._osize[index], (0, 0, 0))
        bg = Image.new('RGB', self._osize[index], (255, 255, 255))
        bg.paste(fg, mask=mask)
        bg.save('./predicts/{:s}'.format(os.path.split(self._impaths[index])[-1]))

    
