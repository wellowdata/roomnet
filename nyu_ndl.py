
#%%
get_ipython().system('git clone https://github.com/DhruvJawalkar/Depth-Map-Prediction-from-a-Single-Image-using-a-Multi-Scale-Deep-Network.git')

import os
os.chdir('Depth-Map-Prediction-from-a-Single-Image-using-a-Multi-Scale-Deep-Network')

  
import torch

from urllib.request import urlretrieve
from torchvision import transforms
from dataset import NYUDataset
from custom_transforms import *
import plot_utils
import model_utils
from nn_model import Net
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

get_ipython().run_line_magic('matplotlib', 'inline')
import torch.nn.functional as F


#%%
bs = 8
sz = (320,240)
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
mean, std = torch.tensor(mean), torch.tensor(std)
unnormalize = UnNormalizeImgBatch(mean, std)

tfms = transforms.Compose([
    ResizeImgAndDepth(sz),
    RandomHorizontalFlip(),
    ImgAndDepthToTensor(),
    NormalizeImg(mean, std)
])


#%%
print(os.getcwd())


#%%
nyu_depth_data_labeled = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_data_labeled.mat'

download_path = os.path.join(os.getcwd(), 'data/nyu_depth_data_labeled.mat')
urlretrieve(nyu_depth_data_labeled,download_path)

nyu_depth_v2_labeled = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
download_path = os.path.join(os.getcwd(), 'data/nyu_depth_v2_labeled.mat')
urlretrieve(nyu_depth_data_labeled,download_path)


#%%
ds = NYUDataset('data/', tfms)
dl = torch.utils.data.DataLoader(ds, bs, shuffle=True)


#%%
ds[0][0].shape


#%%
i = 1
plot_utils.plot_image(model_utils.get_unnormalized_ds_item(unnormalize, ds[i]))


#%%
model = Net()
model.to(device)


#%%
model_ckpt_url = 'https://storage.googleapis.com/ucl-interior-desing/nyu_dnl_pretrained.ckpt'
download_path = os.path.join(os.getcwd(), 'data/all-scales-trained.ckpt')
urlretrieve(model_ckpt_url,download_path)


#%%
model.load_state_dict(torch.load('data/nyu_dnl_pretrained.ckpt', map_location="cpu"))


#%%
model.train()
n_epochs = 0
lr = 0.0000005
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)

total_steps = 0
for e in range(n_epochs):
    for batch, labels in dl:
        optimizer.zero_grad()
        
        batch = batch.to(device)
        labels = labels.to(device)
        
        preds = model(batch)
        loss = model_utils.depth_loss(preds, labels) 
        
        loss.backward()
        optimizer.step()
        
        total_steps +=1
                                                       
        model_utils.print_training_loss_summary(loss.item(), total_steps, e+1, n_epochs, len(dl))


#%%
get_ipython().run_cell_magic('time', '', 'with torch.no_grad():\n    model.eval()\n    img, depth = iter(dl).next()\n    preds = model(img.to(device))')


#%%
plot_utils.plot_model_predictions_on_sample_batch(images=unnormalize(img), depths=depth, preds=preds.squeeze(dim=1), plot_from=0)


#%%



