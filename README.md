# Multi-Scale Efficient Attention Net

The official PyTorch code implementation for ICITR 2023 submission: "Multi-Scale Efficient Attention Net"

This implementation is based on open-reid and APNet (https://github.com/CHENGY12/APNet).


## Introduction
The goal of the person re-identification is to retrieve the specific person across different camera views and times, which has been widely applied in many applications, such as suspect tracking, video surveillance, intelligent security, smart city, and so on. With the advancement of deep learning and machine learning, person re-identification has rapidly growth in recent years. Despite the advancements made in recent times, person re-identification is still a challenging problem due to large intra-class variance caused by occlusions or cluttered backgrounds and pose variations.
In recent times, numerous research endeavors have turned to attention-based designs to tackle the mentioned challenges in person re-identification. This approach aims to enhance the discriminative features while mitigating interferenceMost of  These methods learn to explore salient regions in the global image, which can be formulated as a salient detection task. However, detecting the salient regions with the attention model is confronted as a challenging problem to jointly capture both coarse and fine-grained clues since the focus varies as the image scale changes. There exist minimal works that jointly learn the attentions under different scales. Also, these models are demanding high computational resources, making it difficult to use in real-world applications. To tackle this issue, in this paper, we introduce a lightweight model named Multi-Scaled Efficient Attention Net (MSEA Net) while preserving accuracy.

We validate our method in Market1501 dataset


![image]https://github.com/dulithchinthaka/person_reid_multi_scale/blob/main/images/Slide1.PNG

![image]
https://github.com/dulithchinthaka/person_reid_multi_scale/blob/main/images/pipeline.png

## Requirements
- Python 3.6+
- PyTorch 1.5+
- CUDA 10.0+

Configuration other than the above setting is untested and we recommend to follow our setting.

To build all the dependency, please follow the instruction below.
```
conda create -n apnet python=3.7 -y
conda activate apnet
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
conda install ignite -c pytorch
git clone https://github.com/CHENGY12/APNet.git
pip install -r requirements.txt
```

To download the pretrained ResNet-50 model, please run the following command in your python console:
```
from torchvision.models import resnet50
resnet50(pretrained=True)
```
The model should be located in RESNET_PATH=```/home/YOURNAME/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth``` or ```/home/YOURNAME/.cache/torch/checkpoints/resnet50-19c8e357.pth```

### Downloading
- Market-1501
- DukeMTMC-reID 
- MSMT17
### Preparation
After downloading the datasets above, move them to the `Datasets/` folder in the project root directory, and rename dataset folders to 'market1501', 'duke' and 'msmt17' respectively. I.e., the `Datasets/` folder should be organized as:
```
|-- market1501
    |-- bounding_box_train
    |-- bounding_box_test
    |-- ...
|-- duke
    |-- bounding_box_train
    |-- bounding_box_test
    |-- ...
|-- msmt17
    |-- bounding_box_train
    |-- bounding_box_test
    |-- ...
```

## Usage
### Training
Change the PRETRAIN_PATH parameter in configs/default.yml to your RESNET_PATH
To train with different pyramid level, please edit LEVEL parareter in configs/default.yml
```
sh train.sh
```
### Evaluation
```
sh test.sh
```

