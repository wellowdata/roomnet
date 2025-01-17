# Mark Wells

Trying to replicate Roomnet for use in my CSML Masters thesis.

Original README below:

# roomnet
This is a tensorflow implementation of room layout paper: [RoomNet: End-to-End Room Layout Estimation](https://arxiv.org/pdf/1703.06241.pdf).

New: You can find pre-trained model [here](https://drive.google.com/open?id=1tyw1fmSfd8LvItCOJrOWMMo7Kpjb7l4S) and sample.npz for [get_res.py](https://github.com/GitBoSun/roomnet/blob/master/roomnet/get_res.py) [here](https://drive.google.com/open?id=1djs4bEBr2XRxzsQVES02siSTnRXvPBz9)

**Note**: This is a simply out-of-interest experiemnt and I cannot guarantee to get the same effect of the origin paper.

## Network
![Roomnet network Architecture](https://github.com/GitBoSun/roomnet/blob/master/images/net.png)
Here I implement two nets: vanilla encoder-decoder version and 3-iter RCNN refined version. As the author noted, the latter achieve better results.

## Data
I use [LSUN dataset](https://www.yf.io/p/lsun) and please download and prepare the RGB images and get a explorationo of the .mat file it includs because they contain layout type, key points and other information.
Here I simply resize the image to (320, 320) with cubic interpolation and do the flip horizontally. (**Note**: When you flip the image, the order of layout key points should also be fliped.) You can see the preparation of data in [prepare_data.py](https://github.com/GitBoSun/roomnet/blob/master/roomnet/prepare_data.py)

## Pre-requests:
You need to install tensorflow>=1.2, opencv, numpy, scipy and other basic dependencies.

## How to use:
Training: 
```
python main.py --train 0 --net vanilla (or rcnn) --out_path path-to-output 
```
Testing:
```
python main.py --test 0 --net vanilla (or rcnn) --out_path path-to-output 
```
## Modifications
1. Classification loss: The author only use loss on the ground truth label while I consider whole classes and I use the common cross entropy loss.
2. Layout loss: I split the layout map to 3 level according to the numerical value which means the foreground and background. The bigger the value is, the larger weight its corresponding loss takes.
3. Upsampling layer: Since I don't find upsampling operation that remember the pooling positions and reproject those back, I simply use the conv_transpose operation in tf.

## Some Results:
### RCNN:
![RCNN Results](https://github.com/GitBoSun/roomnet/blob/master/images/rcnn.png)
### Vanilla
![Vanilla Results](https://github.com/GitBoSun/roomnet/blob/master/images/vanilla.png)
