# pytorch-reconet
This is PyTorch implementation of
 "[ReCoNet: Real-time Coherent Video Style Transfer Network](https://arxiv.org/abs/1807.01197)" paper. 

This model allows to perform style transfer on videos in real-time and preserve temporal consistency between frames.   

## Training
To train a model:

1. Run `python ./data/download_data.py` to download data.
This may take about a day and you need to have >1TB of free space on disk.
You will also need [aria2](https://aria2.github.io/) installed
2. Install python dependencies via `pip install -r requirements.txt`
3. Run `python train.py style_image.jpg` to train model with style from some `style_image.jpg`. 
This script supports several additional arguments that you can find using `python train.py -h`

## Inference

There are two options for inference:

1. There is a programming interface in `lib.py` file.
It contains `ReCoNetModel` class that provides `run` method
that accepts a batch of images as 4-D uint8 NHWC RGB numpy tensor and stylizes it
2. There is a `style_video.py` file to style videos. Run it as
`python style_video.py input.mp4 output.mp4 model.pth`. It also supports some additional arguments.
Note that you will need `ffmpeg` to be installed on your machine to run this script

Pre-trained on `./styles/mosaic_2.jpg` model can be downloaded from here:  
https://drive.google.com/open?id=1MUPb7qf3QWEixZ6daGGI4lVFGmQl0qna 
 
Example video with this model:  
https://youtu.be/rEJrNL_2Lfs
 
## Bubble Artifacts

Training model as described in paper leads to bubble artifacts

![Bubble artifacts](https://github.com/EmptySamurai/pytorch-reconet/blob/master/bubble_artifacts.jpg?raw=true)

This issue was addressed in [StyleGAN2 paper](https://arxiv.org/abs/1912.04958) by NVIDIA team. 
They discovered that artifacts appear because of Instance Normalization.
They also proposed a novel normalization method, but unfortunately it doesn't work good with ReCoNet architecture —
either style and content losses didn't converge or some blurry artifacts appeared.

Instead of that in this implementation a [Filter Response Normalization with Thresholded Linear Unit](https://arxiv.org/abs/1911.09737) can be used.
It acts similar to Instance Normalization but preserves mean values in some sense.
This normalization leads to the same results as original architecture, but lacks bubble artifacts.
Every script and class supports `frn` argument that enables Filter Response Normalization instead of Instance Normalization and also replaces ReLU by TLU.

Pre-trained on `./styles/mosaic_2.jpg` model with FRN can be downloaded from here:    
https://drive.google.com/open?id=1T7P5w_V5cMumeEoXs3WFituiiVGhGb3H

## Notes

1. In this implementation loss weights differ from ones in the paper,
since weights in the paper didn't work. This is probably 
due to different image scale and losses normalization constants
2. Testing using MPI Sintel Dataset is not implemented 
