
# 3D Hand Pose Estimation from Single RGB Images Using Prior-knowledge and Mesh Supervision

This repository is the official implementation of [3D Hand Pose Estimation from Single RGB Images Using Prior-knowledge and Mesh Supervision](https://openreview.net/pdf?id=5-BC94Xd9f6). 

![alt Network Architecture](https://github.com/sundigang/PMSNet/raw/main/fig_netarch.png)




## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Evaluation

To evaluate my model on images, you should copy images into "PMSNet/image/input" and then run:

```eval
python eval.py --model-file mymodel.pth
```
evaluation results will be output to "PMSNet/image/output"


## Pre-trained Models

You can download pretrained models here:

- [PMS_on_inter_frei.pth](https://www.dropbox.com/s/3yj02i77lz6s6k9/PMS_on_inter_frei.pth?dl=0) trained on InterHand2.6M and FreiHAND datasets.
- [PMS_on_inter_frei_stb.pth](https://www.dropbox.com/s/59sq4atntmixeb8/PMS_on_inter_frei_stb.pth?dl=0) trained on InterHand2.6M, FreiHAND, and STB datasets.
- [PMS_on_inter_frei_rhd.pth](https://www.dropbox.com/s/3d4zr6obfqupnh4/PMS_on_inter_frei_rhd.pth?dl=0) trained on InterHand2.6M, FreiHAND, and RHD datasets.



