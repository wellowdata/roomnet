
#%%



#%%
get_ipython().system('git clone https://github.com/ialhashim/DenseDepth.git')

#!cd DenseDepth; ls
  
import tempfile
import tensorflow as tf

from urllib.request import urlretrieve


import os
import glob
import argparse

import os
#print(os.getcwd())
os.chdir('DenseDepth')

# Kerasa / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from loss import depth_loss_function
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt


#%%
model_dir = tempfile.mkdtemp()

#pretrained weight locations
kitti_url = r'https://s3-eu-west-1.amazonaws.com/densedepth/kitti.h5'
nyu_url = r'https://s3-eu-west-1.amazonaws.com/densedepth/nyu.h5'

tf.gfile.MakeDirs(model_dir)
download_path = os.path.join(model_dir, 'nyu.h5')

urlretrieve(nyu_url,download_path)


#%%
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}


# Load model into GPU / CPU
model = load_model(download_path, custom_objects=custom_objects, compile=False)


#%%
img_url= 'https://lh3.googleusercontent.com/RP4b_5XbTTZ7V-l5yVpbDZYNLISMfl2DOv6UaUKWbAXG8nU4MdpT9z76l8QbhEYPdCdy4e04z3KNr3ARYXeFnitQpb0zzVchCYV_faUoGudFvWati1z3TVMczN__F7zlUesbZccXvWqVO_7DihNW84zLr6eKiEtHxdnip5ShmilpVnxRfCkbtSs2gsw4NGu1ejvAn5DcumBNc0JZz2xStBfO22m9SWZTO8ZYkyfuWRA2rIGqvhZYG51wbf-41SfnBA72c-jraAVeaNOcscWYk_tUOabCbRKu-8A_dulnD6fTFU1Q1WtcyM6VMNns89Hwiu3NNPdQtNMSgE02Y7oA2MV2cWofwWVMRoiAbrtAPOZ2ekJxBNrKcE0RTq_R4-eKF5QRVLjST4HSUVrnknzj5Ev67S0_7owXjLnu3VS-JzIBq0XtUUcZilnqMMgEvxp5UYgRrkZplUos4QF8yHNx5z-VczMQBdW_AzhL0TgGpZz7X6_y9IZXzd_hcWAN-2c7hMcGFBJN0oWZUR00MEwsM1H9lfa4rAbkfU7ohH5KV1jTkJ8ZzdvsVRyevL67mQ7KgRoS9Q_ePBJghotp-4P_-WCUsAQKMLEPllJfglqlmGPnLVxXi8gbgP0mitGXOyjdU7EBCzxK4yslMN_nLrXDckPU3NtJJfKx=w596-h794-no'

from six.moves import urllib
from PIL import Image
from io import BytesIO

f = urllib.request.urlopen(img_url)
jpeg_str = f.read()
## only works if you're authenticated by google drive
original_im = Image.open(BytesIO(jpeg_str))

# Input images
inputs = load_images( glob.glob(original_im) )
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))


#%%
ADE20k_url = 'https://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip'
#!wget https://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip


#%%
## takes a long time
#!unzip ADE20K_2016_07_26.zip


#%%
for picture in os.listdir('ADE20K_2016_07_26/images/validation/a/abbey'):
  inputs = load_images( glob.glob(picture) )
  print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

  # Compute results
  outputs = predict(model, inputs)

  # Display results
  viz = display_images(outputs.copy(), inputs.copy())
  plt.figure(figsize=(10,5))
  plt.imshow(viz)
  plt.show()


#%%
tf.reset_default_graph()
  


#%%
path = 'ADE20K_2016_07_26/images/validation/h/hospital_room/ADE_val_00001426.jpg'
inputs = load_images( glob.glob(path) )
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = predict(model, inputs)

# Display results
viz = display_images(outputs.copy(), inputs.copy())
plt.figure(figsize=(10,5))
plt.imshow(viz)
plt.show()


#%%



