# SSL DINO YOLO: Self-Supervised Learning for YOLO with DINO

This repository implements a self-supervised learning (SSL) approach for training the YOLO object detection model using **DINO** (Self-Supervised Learning with Contrastive Loss). The goal of the project is to improve YOLO's performance by leveraging self-supervised learning techniques to pretrain the backbone of the model, followed by fine-tuning on a labeled dataset.

## Key Features:
- **DINO Self-Supervised Learning**: Utilizes DINO's contrastive learning approach to pretrain the YOLO backbone. The DINO projection head is used to map features into a lower-dimensional space, facilitating improved feature learning.
- **Fine-Tuning with YOLO**: After pretraining, the model is fine-tuned on a labeled object detection dataset, improving performance in tasks like classification and bounding box prediction.
- **Custom Dataset Support**: Supports training on custom datasets (e.g., AITOD) by specifying dataset paths and labels in a `data.yaml` configuration file.
- **Easy Setup**: The project is designed to be easy to set up and run. Simply modify the dataset paths in the YAML file, and you're ready to train your model.
- **Compatibility**: Compatible with various image sizes and can be used on both CPU and GPU (CUDA-enabled devices).

## How It Works:
1. **Pretraining**: The backbone of YOLO is pretrained using the DINO projection head, which learns representations without requiring labeled data.
2. **Fine-Tuning**: After pretraining, the model is fine-tuned on a labeled dataset (e.g., VOC or custom datasets), using standard supervised learning.
3. **Object Detection**: The model is used to detect objects in images and videos, providing both class predictions and bounding boxes.

## Requirements:
- Python 3.x
- PyTorch
- Ultralytics YOLOv5
- Lightly

## Installation:
To get started, clone this repository and install the dependencies:
```bash
git clone https://github.com/shirbenami/ssl-dino-yolo.git
cd ssl-dino-yolo
pip install -r requirements.txt
```

## Usage:
1. **Pretraining with SSL DINO**:
   - In the `main.py` file, the model is pretrained using **Self-Supervised Learning (SSL)** with the **DINO** approach on an **unlabeled dataset**.
   
2. **Supervised Training**:
   - You can compare the SSL-trained model with a **Supervised** YOLO model by running the `supervised.py` script, which trains YOLO with labeled data.

3. **Fine-Tuning**:
   - After performing supervised training, you can fine-tune the SSL-trained model using the `fine_tune.py` script. This script loads the pretrained weights from the model trained in `main.py` and fine-tunes it on your labeled dataset.
   
4. **Comparison**:
   - The fine-tuned model can then be compared with the results of the supervised model to analyze and evaluate the impact of self-supervised learning on YOLO's performance.
   


   
