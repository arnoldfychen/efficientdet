# EfficientDet: Scalable and Efficient Object Detection

## Improvements
   I had ever tried other version implementation of EfficientDet, results seemed very bad,(BTW,I'm sure my dataset is OK as training and testing were successfully done with
 different network, such as Faster-RCNN(ZF,VGG) and Cascade-RCNN (Resnet101,ResneXt101,HRnet) and CenterNet, and results were considerably good), that code implementation 
seems something wrong, and finally came across Signatrix's this code implementation, training and testing results are OK with my own dataset, but I think there are some flaws 
that need to patched up, so, I forked Signatrix's project and had a try to revamp some code lines, I have done the following improvements in my this branch:

  (1)As of now,the original version only works well with network efficientdet-d0 and efficientdet-d1, errors will happen with other level of Efficientdet, I add some changes to make
     d2-d7 also supported, now weights can be correctly loaded form Efficientnet b2-b7 pretrained model file for Efficientnet backbone, and training with Efficientdet d2-d7 works fine.

  (2)The original code downloads the corresponding Efficientnet pretrained model files from https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/, 
     this is easy to maintain, but if you are in a bad network with poor network speed, you will suffer the slow direct downloading without speedup tool. We can download those
     pretrained model files in advance with speedup tools to local directory and load the corresponding weights locally, this will improve user experience very much, I add some
     code to support loading pretrained weights locally, you can set the loading mode with the command line argument --remote-loading = True or False, when using the default
     loading mode,i.e., loading weights locally from ./pretrained_models, you should make the directory ./pretrained_models in advance under the root directory of source code, and  download
     the pretrained model files from https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/ into it.

  (3)Signatrix's code doesn't support resuming traing from a epoch where training stopped unexpectedly, this is of a little bitter if you have run the training for long time and stopped 
     occasionally due to unexpected disturbing such as power cut, so,I added some statements to support resuming.    

## Usage 

   You can set arguments on the command line or in train.py, the new arguments added by me are the following ones:
      --save_interval     default=10,    the number of epoches between two operations for saving weights
      --backbone_network  default='efficientnet-b0', the value is one of efficientnet-b0/efficientnet-b1/.../efficientnet-b7
      --remote_loading    default=False, if this option is enabled, it will download and load the backbone weights from https://github.com/lukemelas/EfficientNet-PyTorch, otherwise,
                                         load the weights locally from ./pretrained_models
      --advprop           default=False, if this option is enabled, the adv_efficientnet_b* backbone will be used instead of efficientnet_b*
      --resume            action='store_true',if resume training from the last model file saved by the last stopped training
      --start_epoch       default=0,     the start_epoch where you restart training by resuming from a model generated recently 

   I think a very important argument is batch_size you need to tune, the default value is 32, OOM error will often happen with many type of GPUs unless your GPU has very big memory,
   I trained EfficientDet-d7 with 4 RTX 2080TI GPUs, whose memory is 11G per one, if I set batch_size greater than 3, Out-of-Memory error always occurred.
    
    Comannd Examples:
    1) python train.py
    2) python train.py --batch_size 3
    3) python train.py --save_interval 50 --backbone_network 'efficientnet-b7' --resume   --start_epoch 101 
    
## Notes

    1)I made changes with efficientnet_pytorch's model.py and utils.py to support loading weights locally, and integrate them here, so, you don't need to install
      the efficientnet_pytorch package as prerequisite any more.  

    2)I made changes in src/dataset.py according to my coco dataset, its num_class is 1 and paths are:
  ```
    COCO
    ├── annotations
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    │
    ├── train2017
    └── val2017
  ```
   You need to change the code in dataset.py as per your dataset.

    

# The following is the original README written by Signatrix GmbH
## Introduction

Here is our pytorch implementation of the model described in the paper **EfficientDet: Scalable and Efficient Object Detection** [paper](https://arxiv.org/abs/1911.09070) (*Note*: We also provide pre-trained weights, which you could see at ./trained_models) 
<p align="center">
  <img src="demo/video.gif"><br/>
  <i>An example of our model's output.</i>
</p>


## Datasets


| Dataset                | Classes |    #Train images      |    #Validation images      |
|------------------------|:---------:|:-----------------------:|:----------------------------:|
| COCO2017               |    80   |          118k         |              5k            |

Create a data folder under the repository,

```
cd {repo_root}
mkdir data
```
  
- **COCO**:
  Download the coco images and annotations from [coco website](http://cocodataset.org/#download). Make sure to put the files as the following structure:
  ```
  COCO
  ├── annotations
  │   ├── instances_train2017.json
  │   └── instances_val2017.json
  │── images
      ├── train2017
      └── val2017
  ```
  
## How to use our code

With our code, you can:

* **Train your model** by running **python train.py**
* **Evaluate mAP for COCO dataset** by running **python mAP_evaluation.py**
* **Test your model for COCO dataset** by running **python test_dataset.py --pretrained_model path/to/trained_model**
* **Test your model for video** by running **python test_video.py --pretrained_model path/to/trained_model --input path/to/input/file --output path/to/output/file**

## Experiments

We trained our model by using 3 NVIDIA GTX 1080Ti. Below is mAP (mean average precision) for COCO val2017 dataset 

|   Average Precision   |   IoU=0.50:0.95   |   area=   all   |   maxDets=100   |   0.314   |
|-----------------------|:-------------------:|:-----------------:|:-----------------:|:-------------:|
|   Average Precision   |      IoU=0.50     |   area=   all   |   maxDets=100   |   0.461   |
|   Average Precision   |      IoU=0.75     |   area=   all   |   maxDets=100   |   0.343   |
|   Average Precision   |   IoU=0.50:0.95   |   area= small   |   maxDets=100   |   0.093   |
|   Average Precision   |   IoU=0.50:0.95   |   area= medium  |   maxDets=100   |   0.358   |
|   Average Precision   |   IoU=0.50:0.95   |   area=  large  |   maxDets=100   |   0.517   |
|     Average Recall    |   IoU=0.50:0.95   |   area=   all   |   maxDets=1     |   0.268   |
|     Average Recall    |   IoU=0.50:0.95   |   area=   all   |   maxDets=10    |   0.382   |
|     Average Recall    |   IoU=0.50:0.95   |   area=   all   |   maxDets=100   |   0.403   |
|     Average Recall    |   IoU=0.50:0.95   |   area= small   |   maxDets=100   |   0.117   |
|     Average Recall    |   IoU=0.50:0.95   |   area= medium  |   maxDets=100   |   0.486   |
|     Average Recall    |   IoU=0.50:0.95   |   area=  large  |   maxDets=100   |   0.625   |


## Results

Some predictions are shown below:

<img src="demo/1.jpg" width="280"> <img src="demo/2.jpg" width="280"> <img src="demo/3.jpg" width="280">

<img src="demo/4.jpg" width="280"> <img src="demo/5.jpg" width="280"> <img src="demo/6.jpg" width="280">

<img src="demo/7.jpg" width="280"> <img src="demo/8.jpg" width="280"> <img src="demo/9.jpg" width="280">


## Requirements

* **python 3.6**
* **pytorch 1.2**
* **opencv (cv2)**
* **tensorboard**
* **tensorboardX** (This library could be skipped if you do not use SummaryWriter)
* **pycocotools**
* **efficientnet_pytorch**

## References
- Mingxing Tan, Ruoming Pang, Quoc V. Le. "EfficientDet: Scalable and Efficient Object Detection." [EfficientDet](https://arxiv.org/abs/1911.09070).
- Our implementation borrows some parts from [RetinaNet.Pytorch](https://github.com/yhenon/pytorch-retinanet)
  

## Citation

    @article{EfficientDetSignatrix,
        Author = {Signatrix GmbH},
        Title = {A Pytorch Implementation of EfficientDet Object Detection},
        Journal = {https://github.com/signatrix/efficientdet},
        Year = {2020}
    }
