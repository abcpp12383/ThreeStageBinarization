import cv2
from cv2 import imshow
import numpy as np 
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
import pywt

def main():
    image_dir = '../ThreeStageBinarization/image/'
    rename_dir = '../ThreeStageBinarization/select_image/original_image_rename/'
    gray_dir = '../ThreeStageBinarization/select_image/original_image_gray/'
    turn_img_to_gray(image_dir,rename_dir,gray_dir)
    split_3_colors = '../ThreeStageBinarization/select_image/original_image)split_colors/'
    image_split_3_colors(image_dir,split_3_colors)
    DWT_4_image_dir = '../ThreeStageBinarization/select_image/original_image_DWT_image/'
    resize_LL_image_dir = '../ThreeStageBinarization/select_image/original_imagebig_LL_image/'
    LL_normalized_dir = '../ThreeStageBinarization/select_image/original_image_LL_normalized/'
    split_merge_DWT_0001(image_dir,DWT_4_image_dir,resize_LL_image_dir,LL_normalized_dir)

def turn_img_to_gray(input_image_dir,input_rename_dir,input_gray_dir):
    # os.makedirs(input_image_dir, exist_ok=True)
    os.makedirs(input_rename_dir, exist_ok=True)
    os.makedirs(input_gray_dir, exist_ok=True)
    image_pathes = os.listdir(input_image_dir)
    
    i = 0
    for image_path in image_pathes:
        image_name = image_path.split('.')[0]
        image_Filename_Extension = image_path.split('.')[1]
        image = cv2.imread(os.path.join(input_image_dir,image_path))
        cv2.imwrite("%s%s.%s"%(input_rename_dir,i,image_Filename_Extension), image)
        i += 1
    rename_pthes = os.listdir(input_rename_dir)
    for rename_pth in rename_pthes:
        image_rename = rename_pth.split('.')[0]
        image_rename_Filename_Extension = rename_pth.split('.')[1]
        rename_image = cv2.imread(os.path.join(input_rename_dir,rename_pth))
        gray_image = cv2.cvtColor(rename_image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("%s%s.%s"%(input_gray_dir,image_rename,image_rename_Filename_Extension), gray_image)

def image_split_3_colors(input_image_dir,input_split_3_colors):
    os.makedirs(input_split_3_colors, exist_ok=True)
    imagepthSSS = os.listdir(input_image_dir)
    i = 0
    for imagepth in imagepthSSS:
        image_name = imagepth.split('.')[0]
        image_filename_Extension = imagepth.split('.')[1]
        rename_image = cv2.imread(os.path.join(input_image_dir,imagepth))
        RED_COLOR_img, GREEN_COLOR_img, BLUE_COLOR_img = split_RGBThreeChannel(rename_image,i)
        cv2.imwrite('%s%s_red.%s'%(input_split_3_colors,i,image_filename_Extension),RED_COLOR_img)
        cv2.imwrite('%s%s_green.%s'%(input_split_3_colors,i,image_filename_Extension),GREEN_COLOR_img)
        cv2.imwrite('%s%s_blue.%s'%(input_split_3_colors,i,image_filename_Extension),BLUE_COLOR_img)
        i += 1

def merge_RGBThreeChannel(R, G, B):
    img = cv2.merge([B, G, R])
    
    return img
def split_RGBThreeChannel(img,i):
    (B, G, R) = cv2.split(img) # 3 channel
    
    # make all zeros channel
    zeros = np.zeros(img.shape[:2], dtype = np.uint8)

    print("R channel:%s image"%i)
    RED_COLOR = merge_RGBThreeChannel(R=R, G=zeros, B=zeros)
    
    print("G channel:%s image"%i)
    GREEN_COLOR = merge_RGBThreeChannel(R=zeros, G=G, B=zeros)
    
    print("B channel:%s image"%i)
    BLUE_COLOR = merge_RGBThreeChannel(R=zeros, G=zeros, B=B)
    
    return RED_COLOR, GREEN_COLOR, BLUE_COLOR
    
def split_merge_DWT(img,wavelet):
    (B, G, R) = cv2.split(img)
    coeffs_B = pywt.dwt2( B, wavelet )
    coeffs_G = pywt.dwt2( G, wavelet )
    coeffs_R = pywt.dwt2( R, wavelet )
    LL_B, (LH_B, HL_B, HH_B) = coeffs_B
    LL_G, (LH_G, HL_G, HH_G) = coeffs_G
    LL_R, (LH_R, HL_R, HH_R) = coeffs_R
    LL = cv2.merge([LL_B, LL_G, LL_R])
    LH = cv2.merge([HL_B, HL_G, HL_R])
    HL = cv2.merge([LH_B, LH_G, LH_R])
    HH = cv2.merge([HH_B, HH_G, HH_R])
    return LL,HL,LH,HH

def DWT_big_LL( f, wavelet ):
    
	coeffs = pywt.dwt2( f, wavelet )
	LL, (LH, HL, HH) = coeffs
    	
	ROW_1, COL_1 = LH.shape[:2]
	nr1, nc1 = LL.shape[:2]
    
	big_LL_normalized = np.zeros( [nr1 * 2, nc1 * 2], dtype = 'uint8' )
    # LL_normalized_image = np.zeros( [ROW_1])
	# LL (Normalized for Display)
	LL_normalized = np.zeros( [ROW_1, COL_1] )
	cv2.normalize( LL, LL_normalized, 0, 255, cv2.NORM_MINMAX )
	g = np.uint8( LL_normalized[:,:] )

	big_LL_normalized[0:2*nr1,0:nc1*2] = cv2.resize(g,(nr1 * 2, nc1 * 2))

	return big_LL_normalized

    # nr2, nc2 = DWT_4_image.
    # DWT_big_LL_image = np.zeros( [nr1])

def DWT_image_db1( img ):
	
	coeffs = pywt.dwt2( img, 'db1' )
	LL, (LH, HL, HH) = coeffs	
	
	nr1, nc1 = LL.shape[:2]
	g = np.zeros( [nr1 * 2, nc1 * 2], dtype = 'uint8' )
    
	g[0:nr1,0:nc1] = np.uint8(LL)
	g[0:nr1,nc1:2*nc1] = np.uint8(LH) 
	g[nr1:2*nr1,0:nc1] = np.uint8(HL)
	g[nr1:2*nr1,nc1:nc1*2] = np.uint8(HH)	
    
	return g,g[0:nr1,0:nc1]

def split_merge_DWT_0001(input_image_dir,input_DWT_4_image_dir,input_resize_LL,input_LL_normalized_dir):
    os.makedirs(input_DWT_4_image_dir, exist_ok=True)
    os.makedirs(input_resize_LL, exist_ok=True)
    os.makedirs(input_LL_normalized_dir,exist_ok=True)
    image_pathes = os.listdir(input_image_dir)
    i = 0
    for image_path in image_pathes:
        image_Filename_Extension = image_path.split('.')[1]
        image = cv2.imread(os.path.join(input_image_dir,image_path))
        (B, G, R) = cv2.split(image)
        i += 1
        name_color = ["blue", "green" ,"red"]
        for j in name_color:
            for color in (B, G, R):
                image_4 = DWT_image_db1(color)[0]
                cv2.imwrite("%s%s_%s.%s"%(input_DWT_4_image_dir,i,j,image_Filename_Extension), image_4)
                image_LL = DWT_image_db1(color)[1]
                row_LL, column_LL = image_LL.shape[:2]
                image_LL_re = cv2.resize(image_LL,(row_LL*2, column_LL*2))
                cv2.imwrite("%s%s_%s.%s"%(input_resize_LL,i,j,image_Filename_Extension),image_LL_re)
                LL_norm = DWT_image_GO(color)
                cv2.imwrite("%s%s_%s(g1).%s"%(input_LL_normalized_dir,i,j,image_Filename_Extension),LL_norm)

def DWT_big_LL_normalize( img ):
    # img1 = cv2.imread( img)
    coeffs = pywt.dwt2( img, 'db1' )
    LL, (LH, HL, HH) = coeffs	
	
    nr1, nc1 = LL.shape[:2]
    g = np.zeros( [nr1 * 2, nc1 * 2], dtype = 'uint8' )

    # g[0:nr1,0:nc1] = np.uint8(LL)
    # g[0:nr1,nc1:2*nc1] = np.uint8(LH) 
    # g[nr1:2*nr1,0:nc1] = np.uint8(HL)
    # g[nr1:2*nr1,nc1:nc1*2] = np.uint8(HH)	
    nr1, nc1 = img.shape[:2]
    g = np.zeros( [nr1 , nc1 ], dtype = 'uint8' )
    LL_normalized = np.zeros( [nr1, nc1] )
    cv2.normalize(LL,LL_normalized, 0, 255)
    g = ( LL_normalized[:,:] )

    return g  
    
def DWT_image_GO( f ):
	nr, nc = f.shape[:2]
	coeffs = pywt.dwt2( f,'db1' )
	LL, (LH, HL, HH) = coeffs	
	
	nr1, nc1 = LL.shape[:2]
	g = np.zeros( [nr1 * 2, nc1 * 2], dtype = 'uint8' )

	# LL (Normalized for Display)
	LL_normalized = np.zeros( [nr1, nc1] )
	cv2.normalize( LL, LL_normalized, 0, 255, cv2.NORM_MINMAX )
	g = np.uint8( LL_normalized[:,:] )
	g321 = cv2.resize(g,(nr1*2, nc1*2))
	
	return g321

if __name__ == '__main__':
    main()