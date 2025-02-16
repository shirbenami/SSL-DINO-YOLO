# SSL DINO YOLO: Self-Supervised Learning for YOLO with DINO

This repository implements a self-supervised learning (SSL) approach for training the YOLO (You Only Look Once) object detection model using **DINO** (Self-Supervised Learning with Contrastive Loss). The goal of the project is to improve YOLO's performance by leveraging self-supervised learning techniques to pretrain the backbone of the model, followed by fine-tuning on a labeled dataset.

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
git clone https://github.com/your-username/ssl-dino-yolo.git
cd ssl-dino-yolo
pip install -r requirements.txt
```

## Usage:
1. Prepare your dataset and modify the `data.yaml` file to specify the paths to your training, validation, and test datasets.
2. Run the training script:
   ```bash
   python main.py
   ```
3. The model will be trained using DINO's self-supervised learning approach, followed by fine-tuning on your labeled dataset.

   
