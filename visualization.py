import numpy as np
import torch
import torchvision
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import matplotlib.pyplot as plt
import os

# Define the model in the same way it was previously defined
model = torchvision.models.vgg16(weights='DEFAULT')
model.classifier[-2] = torch.nn.Linear(4096, 53)
model.classifier[-1] = torch.nn.Softmax(1)
model.classifier[2] = torch.nn.Dropout(p=0.1)

# Load the state (the weights) of the model
model.load_state_dict(torch.load('transfer_vgg_cards_conv_train_100.pt'))

# Set model in evaluation mode
model.eval()
# Instantiate cam object
cam = EigenCAM(model=model, target_layers=[model.features[-1]])

# Input card path
card_type = input('Card class: ')
card_path = input('Card path (class-specific): ')


input_image = cv2.imread(os.path.join('dataset_cards', 'train', card_type, card_path))
input_tensor = torch.tensor(np.expand_dims(np.moveaxis(input_image/255.0, 2, 0).astype(np.float32), 0))

# Get output image from cam
out = cam(input_tensor)[0]

# plot image
image_toplot = show_cam_on_image(input_image/255.0, out, use_rgb=True)
plt.imshow(image_toplot)
plt.xticks([])
plt.yticks([])
plt.show()
