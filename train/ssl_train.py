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

def ssl_train(model, dataloader, criterion, optimizer, EPOCHS):
    print("Starting Training")
    epochs = EPOCHS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for epoch in range(epochs):
        total_loss = 0
        momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)
        # Initialize the progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for batch in pbar:
            views = batch[0]
            update_momentum(model.student_backbone, model.teacher_backbone, m=momentum_val)
            update_momentum(model.student_head, model.teacher_head, m=momentum_val)
            views = [view.to(device) for view in views]
            global_views = views[:2]
            teacher_out = [model.forward_teacher(view) for view in global_views]
            student_out = [model.forward(view) for view in views]
            loss = criterion(teacher_out, student_out, epoch=epoch)
            total_loss += loss.detach()
            loss.backward()
            # We only cancel gradients of student head.
            model.student_head.cancel_last_layer_gradients(current_epoch=epoch)
            optimizer.step()
            optimizer.zero_grad()

            # Update the progress bar with the current batch loss
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.5f}")

    return avg_loss,model