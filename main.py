import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from synthesize import generate_dataset
from model import FCN


# Define dataset split sizes.
train_size = 180
valid_size = 20
n_samples = train_size + valid_size

# Create directories for data and validation visualization storage.
for dir_name in ('data', 'viz_results'):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        
path2imgs = os.path.join('data', 'animal_imgs.dat')
path2masks = os.path.join('data', 'animal_masks.dat')


starttime = time.time()
generate_dataset(n_samples, path2imgs, path2masks)
print("Synthesized {} images and masks in {:.2f} seconds.".format(n_samples, time.time() - starttime))
batch_size = 18
val_size = 2

# Example of loading numpy array data from .dat files.
def load_data(n_samples, path2imgs, path2masks):
    imgsmap = np.memmap(path2imgs, dtype=np.uint8, mode='r+', shape=(n_samples, 256, 256, 3))
    masksmap = np.memmap(path2masks, dtype=np.uint8, mode='r+', shape=(n_samples, 256, 256))
    imgarr = np.array(imgsmap)
    maskarr = np.array(masksmap)
    return imgarr, maskarr

imgarr, maskarr = load_data(n_samples, path2imgs, path2masks)

# Define the device. Training will take a lot longer without a cuda gpu.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Split the loaded image and mask pairs into seperated training and validation tensors.
train_imgs = torch.from_numpy(imgarr[:-valid_size]).permute(0, 3, 1, 2).float().to(device)
train_masks = torch.from_numpy(maskarr[:-valid_size]).long().to(device)

valid_imgs = torch.from_numpy(imgarr[-valid_size:]).permute(0, 3, 1, 2).float().to(device)
valid_masks = torch.from_numpy(maskarr[-valid_size:]).long().to(device)

# A function that makes and saves a visualization displaying the original image,
# the target mask, and our model's predicted mask on a validation sample.
def save_viz(epoch_idx, class_mask):
    # The class_mask will be per-pixel-classification tensor of a validation sample.
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    ax1.imshow(imgarr[sample_idx + train_size])
    ax1.set_title("Original")
    ax2.imshow(maskarr[sample_idx + train_size], vmin=0, vmax=n_species+1, cmap='gist_ncar')
    ax2.set_title("Target")
    ax3.imshow(class_mask, vmin=0, vmax=n_species+1, cmap='gist_ncar')
    ax3.set_title("Predicted")
    
    ax1.tick_params(labelbottom=False, labelleft=False)
    ax2.tick_params(labelbottom=False, labelleft=False)
    ax3.tick_params(labelbottom=False, labelleft=False)
    
    fig.savefig('viz_results/viz_{}.png'.format(epoch_idx), bbox_inches='tight')
    plt.close(fig)

# Instantiate the model and its hyperparameters.
n_species = 4
n_epochs = 501
model = FCN(n_classes = n_species, img_size=256).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.NLLLoss()

# Our main training loop. 
for epoch_i in range(n_epochs):
    # Training consists of just these 4 lines.
    loss = criterion(model(train_imgs), train_masks)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Visualize validation performance every 50 epochs.
    if epoch_i % 50 == 0:
        print("Epoch: {}/{}, Loss: {:.4f}".format(epoch_i+1, n_epochs, loss.item()))
        with torch.no_grad():
            sample_idx = np.random.randint(valid_size)  # Get a random validation sample index.
            class_preds = model(valid_imgs[sample_idx].unsqueeze(dim=0))
            class_mask = torch.max(class_preds[0], dim=0)[1]
            class_mask = class_mask.detach().cpu().numpy().astype(np.uint8)
            save_viz(epoch_i, class_mask)

print("Script execution finished in {:.2f} minutes.".format((time.time() - starttime)/60))
