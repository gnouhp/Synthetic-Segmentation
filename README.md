# Synthetic-Segmentation
This repository contains files that show how I synthetically generate images and segmentation masks with just NumPy, Matplotlib, and PIL. Includes a minimal fully-convolutional network implementation in PyTorch 0.4.0.


## Requirements
Every dependency that I used, such as Python 3.6.5, PyTorch 0.4.0, Matplotlib, PIL, and NumPy were installed through conda.


## Synthetic Image Generation
The ability to efficiently generate image datasets in such a fashion has many more advantages, including sparing the human-capital required to annotate individual pixels for large numbers of images. Also, it enables engineers to customize the image data specifically for task or domain objectives. A drawback to synthetic image generation is that image data may be void of potentially valuable relationships between visual features that can be found in natural images.

I created this repository to demonstrate how I create a dataset of synthetic images with annotated pixels as numpy arrays, as shown below:

<img src="https://github.com/gnouhp/Synthetic-Segmentation/blob/master/repo_images/gray0.png" width="300">   <img src="https://github.com/gnouhp/Synthetic-Segmentation/blob/master/repo_images/gray1.png" width="300"> 
<img src="https://github.com/gnouhp/Synthetic-Segmentation/blob/master/repo_images/red0.png" width="300">   <img src="https://github.com/gnouhp/Synthetic-Segmentation/blob/master/repo_images/red1.png" width="300"> 
<img src="https://github.com/gnouhp/Synthetic-Segmentation/blob/master/repo_images/green0.png" width="300">   <img src="https://github.com/gnouhp/Synthetic-Segmentation/blob/master/repo_images/green1.png" width="300"> 
<img src="https://github.com/gnouhp/Synthetic-Segmentation/blob/master/repo_images/blue0.png" width="300">   <img src="https://github.com/gnouhp/Synthetic-Segmentation/blob/master/repo_images/blue1.png" width="300">


## Learning Semantic Segmentation w/a Fully Convolutional Network (from scratch)
I achieved decent performance with a minimal, 3-layer FCN. PyTorch version 0.4.0 and above now includes loss functions that can calculate per-pixel classification losses. I took advantage of this new feature and it really simplifies my training implementation (many older examples on GitHub write custom PyTorch Module classes to calculate the per-pixel losses).

As a note on deep fully convolutional nets, it is usually recommended to first downsample feature maps and then upsample them with transpose convolutions, but for this trivial toy task, I found that this slowed down my training by adding unnecessary depth to my shallow network.

Here is an example of the progression of model's learning on validation samples (segmentation images are captioned with the epoch), after about 10 minutes of training on my gpu-enabled laptop:

![Epoch 0](https://github.com/gnouhp/Synthetic-Segmentation/blob/master/repo_images/viz_0.png)
![Epoch 150](https://github.com/gnouhp/Synthetic-Segmentation/blob/master/repo_images/viz_150.png)
![Epoch 300](https://github.com/gnouhp/Synthetic-Segmentation/blob/master/repo_images/viz_300.png)
![Epoch 450](https://github.com/gnouhp/Synthetic-Segmentation/blob/master/repo_images/viz_450.png)
![Epoch 500](https://github.com/gnouhp/Synthetic-Segmentation/blob/master/repo_images/viz_500.png)

The resolution of the model's class predictions continues to improve with more training epochs, but 500 iterations of gradient descent was sufficient for the purpose of this toy segmentation task. 


## Image Copyright
The 7 animal images used in this dataset are from [Pixabay](https://pixabay.com/en/), and can also be found through Google Advanced Image search marked as free for commerical or personal use without attribution. Thank you to the original artists.
