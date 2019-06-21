
#%%
import os

from urllib.request import urlretrieve

get_ipython().system('git clone https://github.com/sunset1995/HorizonNet.git')


#%%
get_ipython().system('pip install torchfile')


#%%
os.chdir('HorizonNet')
os.listdir()


#%%
#pretrained weight locations
pretrained_model_url = r'https://storage.googleapis.com/ucl-interior-desing/model/resnet50-rnn.pth'
os.mkdir('ckpt')
download_path = os.path.join(os.getcwd(), 'ckpt/resnet50-rnn.pth')

urlretrieve(pretrained_model_url,download_path)


#%%
#%run torch2pytorch_pretrained_weight.py --torch_pretrained ckpt/panofull_joint_box_pretrained.t7 --encoder ckpt/pre_full_encoder.pth --edg_decoder ckpt/pre_full_edg_decoder.pth --cor_decoder ckpt/pre_full_cor_decoder.pth


#%%
test_image_url = r'https://www.watles.com/media/8c84f01c-0c29-4b78-8fc7-2d4d80847e8f/panorama-spa.jpg'
urlretrieve(test_image_url, 'assets/panorama-spa.jpg')


#%%
#%run visual_preprocess.py  --img_glob assert/panorama-spa.jpg --output_dir assert/output_preprocess/
get_ipython().run_line_magic('run', 'preprocess.py --img_glob assets/panorama-spa.jpg --output_dir assets/preprocessed/')
os.listdir('assert/preprocessed/')


#%%
get_ipython().run_line_magic('run', 'inference.py --flip --pth ckpt/resnet50-rnn.pth --img_glob assets/preprocessed/panorama-spa_aligned_rgb.png --output_dir assets/inferenced --visualize --relax_cuboid')


#%%
get_ipython().run_line_magic('run', 'layout_viewer.py --img assets/preprocessed/panorama-spa_aligned_rgb.png --layout assets/inferenced/demo_aligned_rgb.json --ignore_ceiling')


#%%
os.chdir('assets/inferenced')


#%%
os.listdir()


#%%
with open('pano_awlvztwonxadgl.json') as f:
  print(f.readlines())


#%%



