
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
    from lightly.data import LightlyDataset
    from tqdm import tqdm
    from ultralytics import YOLO
    from ultralytics.nn.modules import Conv

    import sys
    print(sys.path)
    from configs.config import GLOBAL_CROP_SIZE, LOCAL_CROP_SIZE, LEARNING_RATE, BATCH_SIZE, EPOCHS
    from models.dino import DINO 
    from models.pool_head import PoolHead

    from train.ssl_train import ssl_train


    yolo = YOLO('models/yolo11n.pt')

    # Only backbone
    yolo.model.model = yolo.model.model[:12]  # Keep first 12 layers

    dummy = torch.rand(2, 3, GLOBAL_CROP_SIZE, GLOBAL_CROP_SIZE) # Create dummy input (2= batch size, 3= channels, GLOBAL_CROP_SIZE= height, GLOBAL_CROP_SIZE= width)
    out = yolo.model.model[:-1](dummy) # Run forward pass only using the first 11 layers

    yolo.model.model[-1] = PoolHead(yolo.model.model[-1].f, yolo.model.model[-1].i, out.shape[1])  # Replace 12th layer with PoolHead

    out = yolo.model(dummy) # Run forward pass on the whole model with the pool head
    input_dim = out.flatten(start_dim=1).shape[1] # Flatten the output and get the number of features
    print("input dim:",input_dim) # Print the number of features

    backbone = yolo.model.requires_grad_() # Set the backbone to require gradients
    backbone.train() # Set the backbone to train mode
    model = DINO(backbone, input_dim) # Create the DINO model (input_dim is the number of features)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    normalize = dict(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0))  # YOLO uses these values
    transform = DINOTransform(global_crop_size=GLOBAL_CROP_SIZE, local_crop_size=LOCAL_CROP_SIZE, normalize=normalize) # Create the DINO transform 

    # we ignore object detection annotations by setting target_transform to return 0
    def target_transform(t):
        return 0

    # create a dataset from the VOC 2012 dataset
    dataset = torchvision.datasets.VOCDetection(
        "datasets/pascal_voc",
        #download=True,
        download=False,
        transform=transform,
        target_transform=target_transform,
    )
    # or create a dataset from a folder containing images or videos:
    dataset = LightlyDataset("datasets/AITOD/crops", transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    criterion = DINOLoss(
        output_dim=2048,
        warmup_teacher_temp_epochs=5,
    )
    # move loss to correct device because it also contains parameters
    criterion = criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    avg_loss,model= ssl_train(model, dataloader, criterion, optimizer,EPOCHS) # Train the model


    # Save the pretrained backbone
    # Load the same model that was used for pretraining
    yolo = YOLO("models/yolo11n.pt")

    # Transfer weights from pretrained model
    yolo.model.load(model.student_backbone)
    
    # Save the model for later use
    yolo.save("models/pretrained.pt")
    