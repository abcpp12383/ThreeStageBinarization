# Three-stage Binarization

> [Three-stage binarization of color document images based on discrete wavelet transform and generative adversarial networks](https://arxiv.org/abs/2211.16098)

### Stage-1 Flowchart
<p align="left">
  <img src="readme_fig/figure_stage_1.jpg" width="480" title="Stage-1">
</p>

### Stage-2 Flowchart
<p align="left">
  <img src="readme_fig/figure_stage_2.jpg" width="480" title="Stage-2">
</p>

### Stage-3 Flowchart
<p align="left">
  <img src="readme_fig/figure_stage_3.jpg" width="480" title="Stage-3">
</p>

## Citation
If you find our paper useful in your research, please consider citing:

**Conference version (accepted by PRICAI 2023)**

    @inproceedings{ju2023ccdwt,
      title={CCDWT-GAN: Generative Adversarial Networks Based on Color Channel Using Discrete Wavelet Transform for Document Image Binarization},
      author={Ju, Rui-Yang and Lin, Yu-Shian and Chiang, Jen-Shiun and Chen, Chih-Chia and Chen, Wei-Han and Chien, Chun-Tse},
      booktitle={Pacific Rim International Conference on Artificial Intelligence},
      pages={186--198},
      year={2023},
      organization={Springer}
    }

**Journal version (accepted by KBS):**

    @article{ju2024three,
      title={Three-stage binarization of color document images based on discrete wavelet transform and generative adversarial networks},
      author={Ju, Rui-Yang and Lin, Yu-Shian and Jin, Yanlin and Chen, Chih-Chia and Chien, Chun-Tse and Chiang, Jen-Shiun},
      journal={Knowledge-Based Systems},
      pages={112542},
      year={2024},
      publisher={Elsevier}
    }
   
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
  
  For Conda users, you can create a new Conda environment using `conda env create -f environment.yaml`

## Dataset
You can download the dataset used in this experiment from [Dropbox](https://www.dropbox.com/scl/fi/rfqshevaq44g81lxkcjtt/Dataset.rar?rlkey=9z1hlrezkq9t99jv5hekuux71&dl=0).

## Usage
* Preprocess
  ```
    python ./Base/image_to_224.py
    python ./Base/image_to_512.py
  ```

* Train the model
  * Stage2
  ```
    python train_stage2.py
  ```
  * Before train left part of Stage3
  ```
    python predict_for_stage3.py
  ```
  * left part of Stage3 (need train predict_for_stage3.py first)
  ```
    python train_stage3.py
  ```
  * right part of Stage3 (independent training)
  ```
    python train_stage3_resize.py
  ```

* Evaluation the model
  ```
    python3 eval_stage3_all.py
  ```

## Related Works

<details><summary> <b>Expand</b> </summary>

* https://github.com/RuiyangJu/Efficient_Document_Image_Binarization
