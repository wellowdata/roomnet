
#%%
import os

from urllib.request import urlretrieve

get_ipython().system('git clone https://github.com/sunset1995/pytorch-layoutnet.git')


#%%
get_ipython().system('pip install torchfile')


#%%
os.chdir('pytorch-layoutnet')


#%%
#pretrained weight locations
pretrained_model_url = r'https://storage.googleapis.com/ucl-interior-desing/model/panofull_joint_box_pretrained.t7'

download_path = os.path.join(os.getcwd(), 'ckpt/panofull_joint_box_pretrained.t7')

urlretrieve(pretrained_model_url,download_path)


#%%
get_ipython().run_line_magic('run', 'torch2pytorch_pretrained_weight.py --torch_pretrained ckpt/panofull_joint_box_pretrained.t7 --encoder ckpt/pre_full_encoder.pth --edg_decoder ckpt/pre_full_edg_decoder.pth --cor_decoder ckpt/pre_full_cor_decoder.pth')


#%%
test_image_url = r'https://www.watles.com/media/8c84f01c-0c29-4b78-8fc7-2d4d80847e8f/panorama-spa.jpg'
urlretrieve(test_image_url, 'assert/panorama-spa.jpg')


#%%
get_ipython().run_line_magic('run', 'visual_preprocess.py  --img_glob assert/panorama-spa.jpg --output_dir assert/output_preprocess/')
os.listdir('assert/output_preprocess/')


#%%
get_ipython().run_line_magic('run', 'visual.py --path_prefix ckpt/pre_full --img_glob assert/output_preprocess/panorama-spa_aligned_rgb.png --line_glob assert/output_preprocess/panorama-spa_aligned_rgb.png --output_dir assert/output')


#%%
get_ipython().system('pip install open3d-python')


#%%
get_ipython().run_line_magic('run', 'visual_3d_layout.py --ignore_ceiling --img assert/output_preprocess/panorama-spa_aligned_rgb.png --layout  assert/output/panorama-spa_aligned_rgb_cor_id.txt')


#%%


