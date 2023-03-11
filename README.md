## [Three-stage binarization of color document images based on discrete wavelet transform and generative adversarial networks](https://arxiv.org/abs/2211.16098)
### Stage-1 Flowchart
<p align="center">
  <img src="Figure/figure_stage_1.jpg" width="480" title="Stage-1">
</p>

### Stage-2 Flowchart
<p align="center">
  <img src="Figure/figure_stage_2.jpg" width="480" title="Stage-2">
</p>

### Stage-3 Flowchart
<p align="center">
  <img src="Figure/figure_stage_3.jpg" width="480" title="Stage-3">
</p>

## Abstract
The efficient segmentation of foreground text information from the background in degraded color document images is a topic of concern. Due to the imperfect preservation of ancient documents over a long period of time, various types of degradation, including staining, yellowing, and ink seepage, have seriously affected the results of image binarization. In this paper, a three-stage method is proposed for image enhancement and binarization of degraded color document images by using discrete wavelet transform (DWT) and generative adversarial network (GAN). In Stage-1, we use DWT and retain the LL subband images to achieve the image enhancement. In Stage-2, the original input image is split into four (Red, Green, Blue and Gray) single-channel images, each of which trains the independent adversarial networks. The trained adversarial network models are used to extract the color foreground information from the images. In Stage-3, in order to combine global and local features, the output image from Stage-2 and the original input image are used to train the independent adversarial networks for document binarization. The experimental results demonstrate that our proposed method outperforms many classical and state-of-the-art (SOTA) methods on the Document Image Binarization Contest (DIBCO) dataset.

## Requirements
* Linux (Ubuntu)
* Python >= 3.6 (Pytorch)
* NVIDIA GPU + CUDA CuDNN

## Installation
* Install [segmentation_models](https://github.com/qubvel/segmentation_models.pytorch)
```
    pip install segmentation-models-pytorch
```
* Install [pytesseract](https://github.com/madmaze/pytesseract)
```
    pip install pytesseract
```
* Download [tesseract data](https://github.com/tesseract-ocr/tessdata_best)
```
    conda env create -f environment.yml
```

## Dataset
* Train: 
  
  DIBCO 2009, H-DIBCO 2010, H-DIBCO 2012, PHIBD, SMADI, Bickley Diary Dataset
  [(Download Link)](https://www.dropbox.com/s/3qwv3jntmgu4rf9/Trainset.zip?dl=0)
  
* Test: 

  DIBCO 2011, DIBCO 2013, H-DIBCO 2014, H-DIBCO 2016, DIBCO 2017, H-DIBCO2018
  [(Download Link)](https://www.dropbox.com/s/54ye0mtdcvqas4o/Testset.zip?dl=0)

## Usage
* Patch per datasets (Part of Stage1)

    Example: 512×512
<p align="center">
<img src="img_512X512/PHIBD_2012_4_h0.png" width="120" title="512_1">
<img src="img_512X512/PHIBD_2012_4_r0.png" width="120" title="512_2">
<img src="img_512X512/PHIBD_2012_4_r1.png" width="120" title="512_3">
<img src="img_512X512/PHIBD_2012_4_r2.png" width="120" title="512_4">
<img src="img_512X512/PHIBD_2012_4_r3.png" width="120" title="512_5">
<img src="img_512X512/PHIBD_2012_4_v0.png" width="120" title="512_6">
</p>

```
    python3 image_to_256.py
    python3 image_to_512.py
```

* Discrete Wavelet Transform (Part of Stage1)

    Example: Blue, Green, Red
<p align="center">
<img src="img_dwt/blue.bmp" width="240" title="blue">
<img src="img_dwt/green.bmp" width="240" title="green">
<img src="img_dwt/red.bmp" width="240" title="red">
</p>

```
    python3 image_dwt_original.py
    python3 image_dwt_256.py
```
* Train the model
** Stage2
```
    python3 train_stage2.py
```
** left part of Stage3 (need train stage2 first)
```
    python3 train_stage3.py
```
** right part of Stage3 (independent training)
    python3 train_stage3_resize.py
```
** Predict
```
    python3 predict_for_stage3.py
```

* Evaluation the model
```
    python3 eval_stage3_all.py
```

## Comparison Experiment of Stage-1 on the H-DIBCO 2014
| Option | Input | GT | FM↑ | p-FM↑ | PSNR↑ | DRD↓ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | \ | \ | 96.50 | 97.50 | 22.08 | 1.01 |
| 2 | \ | DWT(LL) | 96.52 | 97.70 | 22.15 | 0.99 |
| 3 | \ | DWT(LL)+Norm | **96.88** | **98.03** | **22.68** | **0.89** |
| 4 | DWT(LL) | \  | 95.94 | 96.91 | 21.31 | 1.19 |
| 5 | DWT(LL)+Norm | \ | 96.29 | 97.48 | 22.05 | 1.16 |
| 6 | DWT(LL) | DWT(LL) | 96.60 | 97.60 | 22.27 | 0.97 |
| 7 | DWT(LL)+Norm | DWT(LL)+Norm | 96.77 | 97.89 | 22.52 | 0.91 | 

* Option 2 is `method1_NA_LL.py`
* Option 3 is `method2_NA_LLandNorm.py`
* Option 6 is `method3_LL_LL.py`
* Option 7 is `method4_LLandNorm_LLandNorm`

## References
[DocumentBinarization](https://github.com/opensuh/DocumentBinarization)
