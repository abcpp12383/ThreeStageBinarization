import os
import numpy as np
import cv2
import math

def main(argv=None):

    # single image dir
    # image_dir = '../ThreeStageBinarization/image/'
    # mask_dir = '../ThreeStageBinarization/mask/'

    image_dir = '../ThreeStageBinarization/DIBCO/Trainset/image/'
    mask_dir = '../ThreeStageBinarization/DIBCO/Trainset/mask/'

    overlap = 100. / 100.
    imgh = 256
    imgw = 256
    reshape = (imgw, imgh)
    scale_list = [0.75, 1.00, 1.25, 1.50] # sample patches with the scale factor and resize patches to 256 * 256 // 192, 256, 320, 384
    rotation = [0, 3]

    image_save_dir = '../ThreeStageBinarization/DIBCO_resize_256/Trainset/image/'
    mask_save_dir = '../ThreeStageBinarization/DIBCO_resize_256/Trainset/mask/'
    
    # single image dir
    # image_save_dir = '../ThreeStageBinarization/image_patches/256X256/'
    # mask_save_dir = '../ThreeStageBinarization/mask_patches/256X256/'

    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    image_pathes = os.listdir(image_dir)
    for image_path in image_pathes:

        image_name = image_path.split('.')[0]

        # find and read mask file
        if os.path.isfile(os.path.join(mask_dir, image_name) + '.png'):
            mask = cv2.imread(os.path.join(mask_dir, image_name) + '.png', cv2.IMREAD_GRAYSCALE)
        elif os.path.isfile(os.path.join(mask_dir, image_name) + '.bmp'):
            mask = cv2.imread(os.path.join(mask_dir, image_name) + '.bmp', cv2.IMREAD_GRAYSCALE)
        else:
            print('no mask', image_path)
            continue

        # there are few images that have a value (1 ~ 254), bickley image has thin stroke
        mask[mask < 190] = 0
        mask[mask >= 190] = 255

        image = cv2.imread(os.path.join(image_dir + image_path))
        print('processing the image:', image_path)
        # continue

        scale_cnt = 0
        for scale in scale_list:
            # (patches, 256, 256, 3)
            crpW = int(scale * imgw)
            crpH = int(scale * imgh)

            image_patches, _ = get_image_patch_deep(image, crpH, crpW, reshape, overlap=overlap)
            mask_patches, poslist = get_image_patch_deep(mask, crpH, crpW, reshape, overlap=overlap)

            print('get patches: %d' % len(image_patches))
            
            for idx in range(len(image_patches)):
                img_color = image_patches[idx]
                img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

                mask_gray = mask_patches[idx]

                # agumentation
                for k in rotation:
                    img_color_tmp = np.rot90(img_color, k)
                    mask_gray_tmp = np.rot90(mask_gray, k)
                    cv2.imwrite('%s/%s_s%dr%di%d.png' % (image_save_dir, image_name, scale_cnt, k, idx), img_color_tmp)
                    cv2.imwrite('%s/%s_s%dr%di%d.png' % (mask_save_dir, image_name, scale_cnt, k, idx), mask_gray_tmp)

                # exit(1)
            scale_cnt += 1
            # exit(1)

def get_image_patch_deep(image, imgh, imgw, reshape=None, overlap=0.1):

    overlap_wid = int(imgw * overlap)
    overlap_hig = int(imgh * overlap) 

    height, width = image.shape[:2]

    image_list = []
    posit_list = []

    for ys in range(0, height-imgh, overlap_hig):
        ye = ys + imgh
        if ye > height:
            ye = height
        for xs in range(0,width-imgw,overlap_wid):
            xe = xs + imgw
            if xe > width:
                xe = width
            imgpath = image[ys:ye,xs:xe]
            if reshape is not None:
                imgpath = cv2.resize(imgpath, dsize=reshape, interpolation=cv2.INTER_NEAREST)
            image_list.append(imgpath)
            pos = np.array([ys,xs,ye,xe])
            posit_list.append(pos)

    # last coloum 
    for xs in range(0, width-imgw, overlap_wid):
        xe = xs + imgw
        if xe > width:
            xe = width
        ye = height 
        ys = ye - imgh
        if ys < 0:
            ys = 0
            
        imgpath = image[ys:ye,xs:xe]
        if reshape is not None:
            imgpath = cv2.resize(imgpath, dsize=reshape, interpolation=cv2.INTER_NEAREST)
        image_list.append(imgpath)
        pos = np.array([ys,xs,ye,xe])
        posit_list.append(pos)
        
    # last row 
    for ys in range(0, height-imgh, overlap_hig):
        ye = ys + imgh
        if ye > height:
            ye = height
        xe = width
        xs = xe - imgw
        if xs < 0:
            xs = 0
            
        imgpath = image[ys:ye,xs:xe]
        if reshape is not None:
            imgpath = cv2.resize(imgpath, dsize=reshape, interpolation=cv2.INTER_NEAREST)
        image_list.append(imgpath)
        pos = np.array([ys,xs,ye,xe])
        posit_list.append(pos)

    # last rectangle
    ye = height
    ys = ye - imgh
    if ys < 0:
        ys = 0
    xe = width 
    xs = xe - imgw
    if xs < 0:
        xs = 0
        
    imgpath = image[ys:ye,xs:xe]
    if reshape is not None:
        imgpath = cv2.resize(imgpath, dsize=reshape, interpolation=cv2.INTER_NEAREST)
    image_list.append(imgpath)
    pos = np.array([ys,xs,ye,xe])
    posit_list.append(pos)

    return image_list, posit_list

if __name__ == '__main__':
    main()