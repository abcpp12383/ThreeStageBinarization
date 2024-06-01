import os
from torch.utils.data import Dataset
import numpy as np
import cv2
import torch
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from Base.tool_clean import check_is_image
from pywt import dwt2

class Dataset_Return_Four(Dataset):

    def __init__(self, image_dir, mask_dir, base_model_name, encoder_weights, threshold=0.30):
        self.threshold = threshold
        self.preprocess_input = get_preprocessing_fn(base_model_name, pretrained=encoder_weights)

        self.image_pathes = []
        self.mask_pathes = []

        image_pathes = os.listdir(image_dir)
        for image_path in image_pathes:
            if not check_is_image(image_path):
                print('not image', image_path)
                continue

            if not os.path.isfile(os.path.join(mask_dir, image_path)):
                print('no mask', image_path)
                continue

            self.image_pathes.append(os.path.join(image_dir + image_path))
            self.mask_pathes.append(os.path.join(mask_dir + image_path))

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, i):
        image = cv2.imread(self.image_pathes[i])
        mask = cv2.imread(self.mask_pathes[i], cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = np.expand_dims(image_gray, axis=-1)
        
        image_blue = image[:, :, 0]
        image_green = image[:, :, 1]
        image_red = image[:, :, 2]

        #DWT (bd1)
        image_blue_LL,(image_blue_LH,image_blue_HL,image_blue_HH) = dwt2(image_blue,'db1')
        image_green_LL,(image_green_LH,image_green_HL,image_green_HH) = dwt2(image_green,'db1')
        image_red_LL,(image_red_LH,image_red_HL,image_red_HH) = dwt2(image_red,'db1')
        
        nr1, nc1 = image_blue_LL.shape[:2]
        image_blue_LL_norm = np.zeros( [nr1, nc1] )
        cv2.normalize( image_blue_LL, image_blue_LL_norm, 0, 255, cv2.NORM_MINMAX )

        nr2, nc2 = image_green_LL.shape[:2]
        image_green_LL_norm = np.zeros( [nr2, nc2] )
        cv2.normalize( image_green_LL, image_green_LL_norm, 0, 255, cv2.NORM_MINMAX )

        nr3, nc3 = image_red_LL.shape[:2]
        image_red_LL_norm = np.zeros( [nr3, nc3] )
        cv2.normalize( image_red_LL, image_red_LL_norm, 0, 255, cv2.NORM_MINMAX )

        image_blue_LL_norm = np.uint8(cv2.resize(image_blue_LL_norm,(224,224),interpolation=cv2.INTER_CUBIC))
        image_green_LL_norm = np.uint8(cv2.resize(image_green_LL_norm,(224,224),interpolation=cv2.INTER_CUBIC))
        image_red_LL_norm = np.uint8(cv2.resize(image_red_LL_norm,(224,224),interpolation=cv2.INTER_CUBIC))
        
        image_blue_LL_norm = np.expand_dims(image_blue_LL_norm, axis=-1)
        image_green_LL_norm = np.expand_dims(image_green_LL_norm, axis=-1)
        image_red_LL_norm = np.expand_dims(image_red_LL_norm, axis=-1)

        mask_and_blue = np.bitwise_and(image_blue_LL_norm, mask)
        val = np.max(mask_and_blue) * self.threshold
        mask_and_blue[mask_and_blue <= val] = 0
        mask_and_blue[mask_and_blue > val] = 255

        mask_and_green = np.bitwise_and(image_green_LL_norm, mask)
        val = np.max(mask_and_green) * self.threshold
        mask_and_green[mask_and_green <= val] = 0
        mask_and_green[mask_and_green > val] = 255

        mask_and_red = np.bitwise_and(image_red_LL_norm, mask)
        val = np.max(mask_and_red) * self.threshold
        mask_and_red[mask_and_red <= val] = 0
        mask_and_red[mask_and_red > val] = 255

        image_blue_LL_norm = self.preprocess_input(image_blue_LL_norm)
        image_green_LL_norm = self.preprocess_input(image_green_LL_norm)
        image_red_LL_norm = self.preprocess_input(image_red_LL_norm)
        image_gray = self.preprocess_input(image_gray)

        image_blue_LL_norm = torch.from_numpy(image_blue_LL_norm).permute(2, 0, 1).float()
        mask_and_blue = torch.from_numpy(mask_and_blue).permute(2, 0, 1).float() / 255.

        image_green_LL_norm = torch.from_numpy(image_green_LL_norm).permute(2, 0, 1).float()
        mask_and_green = torch.from_numpy(mask_and_green).permute(2, 0, 1).float() / 255.
        
        image_red_LL_norm = torch.from_numpy(image_red_LL_norm).permute(2, 0, 1).float()
        mask_and_red = torch.from_numpy(mask_and_red).permute(2, 0, 1).float() / 255.

        image_gray = torch.from_numpy(image_gray).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.

        return (image_blue_LL_norm, image_green_LL_norm, image_red_LL_norm, image_gray), (mask_and_blue, mask_and_green, mask_and_red, mask)

class Dataset_Return_One(Dataset):

    def __init__(self, image_dir, mask_dir, base_model_name, encoder_weights):
        self.base_model_name = base_model_name
        self.preprocess_input = get_preprocessing_fn(base_model_name, pretrained=encoder_weights)

        self.image_pathes = []
        self.mask_pathes = []

        image_pathes = os.listdir(image_dir)
        for image_path in image_pathes:
            if not check_is_image(image_path):
                print('not image', image_path)
                continue

            if not os.path.isfile(os.path.join(mask_dir, image_path)):
                print('no mask', image_path)
                continue

            self.image_pathes.append(os.path.join(image_dir, image_path))
            self.mask_pathes.append(os.path.join(mask_dir, image_path))

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, i):
        image = cv2.imread(self.image_pathes[i])
        mask = cv2.imread(self.mask_pathes[i], cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)

        image = self.preprocess_input(image, input_space="BGR")
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        mask = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.

        return image, mask
