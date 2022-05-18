
# 3D Hand Pose Estimation from Single RGB Images Using Prior-knowledge and Mesh Supervision

This repository is the official implementation of [3D Hand Pose Estimation from Single RGB Images Using Prior-knowledge and Mesh Supervision](https://openreview.net/pdf?id=5-BC94Xd9f6). 

![alt Network Architecture](https://github.com/sundigang/PMSNet/raw/main/fig_netarch.png)




## Requirements
Install PyTorch >= 1.7.1 and Python >= 3.7.0  
Install requirements:

```setup
pip install -r requirements.txt
```


## Evaluation

To evaluate my model on images, you should  
(1) copy images to "PMSNet/image/input", images will be automatically resized to 256 x 256 pixel  
(2) run eval.py with a pre-trained model:  

```eval
python eval.py --model-file a_pretrained_model.pth
```
Images with 2D/3D pose evaluation results will be output to "PMSNet/image/output"


## Pre-trained Models

You can download pretrained models here:

- [PMS_on_inter_frei.pth](https://www.dropbox.com/s/3yj02i77lz6s6k9/PMS_on_inter_frei.pth?dl=0) trained on InterHand2.6M and FreiHAND datasets with mesh supervision.
- [PMS_on_inter_frei_stb.pth](https://www.dropbox.com/s/59sq4atntmixeb8/PMS_on_inter_frei_stb.pth?dl=0) trained on InterHand2.6M, FreiHAND, and STB datasets without mesh supervision.
- [PMS_on_inter_frei_rhd.pth](https://www.dropbox.com/s/3d4zr6obfqupnh4/PMS_on_inter_frei_rhd.pth?dl=0) trained on InterHand2.6M, FreiHAND, and RHD datasets without mesh supervision.



