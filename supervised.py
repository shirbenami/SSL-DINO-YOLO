if __name__ == '__main__':
    
    import copy
    import torch
    import torchvision
    from torch import nn
    from lightly.loss import DINOLoss
    from lightly.models.modules import DINOProjectionHead
    from lightly.models.utils import deactivate_requires_grad, update_momentum
    from lightly.transforms.dino_transform import DINOTransform
    from lightly.utils.scheduler import cosine_schedule
    from tqdm import tqdm
    from ultralytics import YOLO
    from ultralytics.nn.modules import Conv

    import sys
    from configs.config import GLOBAL_CROP_SIZE, LOCAL_CROP_SIZE, LEARNING_RATE, BATCH_SIZE, EPOCHS
    from models.dino import DINO 
    from models.pool_head import PoolHead

    from train.ssl_train import ssl_train

    

    yolo = YOLO("models/yolo11n.pt")
    
    # Nothing different from your usual process
    results = yolo.train(data="data.yaml", epochs=3, freeze=0, warmup_epochs=0, imgsz=640)
    
    #save the results
    results.save