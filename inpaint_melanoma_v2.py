'''
Performs inpaiting on lung nodules
-Based on the notebook: "inpainting nodules of all patients v17 (from v15) - 2D conv on 3D vol.ipynb"
-Uses 2D convolutions
-This was modified (~line 189)
    image_last = images_generated_all[0] * block_lungs
    image_last = images_generated_all[-1] * block_lungs
-We use num_channels_skip = [128]*5, 
'''
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.optim
from copy import copy, deepcopy
import time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from skimage import measure, morphology
from itertools import groupby, count
import matplotlib.patches as patches
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from torch.autograd import Variable
from scipy.spatial import distance
import sys
sys.path.append('/home/om18/anaconda3/envs/luna/lib/python3.6/site-packages')
sys.path.append('/home/om18/Documents/KCL/Feb 5 19 - Deep image prior/deep-image-prior/')

from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
from utils.inpainting_utils import *

from inpainting_nodules_functions import *
import warnings
from torch.autograd import Variable

from scipy import ndimage
from skimage import filters

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
# parser.add_argument('skip_idx', type=int, help='skip indices already processed')
parser.add_argument('cuda_gpu', type=int, help='gpu to use')
parser.add_argument('channels_input_and_layers', type=str, help='number of channels in the input (noise) and number of channels on each layer')
parser.add_argument('filename', type=str, help='gpu to use')
args = parser.parse_args()

channels_in_lay = [int(i) for i in args.channels_input_and_layers.split('_')]
run_epochs = channels_in_lay[0]
channels_input = channels_in_lay[1]
channels_layers = channels_in_lay[2:]
channels_skip = [32]*len(channels_layers)

## MELANOMA

def rgb2gray(rgb):
    '''https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python'''
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def otsu_thresh_4largest_component(img2):
    val = filters.threshold_otsu(img2)
    mask_otsu_orig = img2<val
    mask_otsu = ndimage.morphology.binary_erosion(mask_otsu_orig, iterations=20)
    mask_otsu = ndimage.morphology.binary_dilation(mask_otsu, iterations=80)
    mask_otsu = ndimage.morphology.binary_fill_holes(mask_otsu)
    labeled_mask, cc_num = ndimage.label(mask_otsu)
    sorted_comp = np.bincount(labeled_mask.flat)
    sorted_comp = np.sort(sorted_comp)[::-1]
    mask_lesions = []
    for i in np.arange(1, np.min([len(sorted_comp), 4])):
        mask_lesions.append((labeled_mask == np.where(np.bincount(labeled_mask.flat) == sorted_comp[i])[0][0]))         
    return mask_lesions

def component_closest_center(img2, masks_lesions):
    y_half, x_half = [i//2 for i in np.shape(img2)]
    y_half_x_half = np.asarray([y_half, x_half])
    ml_closest = masks_lesions[0] # default
    dist_min = 10000
    for i in masks_lesions:
        yy,xx = np.where(i==1) 
        ymed_xmed = np.asarray([np.median(yy), np.median(xx)])
        dist_new = distance.cdist(np.expand_dims(y_half_x_half,0), np.expand_dims(ymed_xmed,0))
        if dist_new < dist_min:
            dist_min = dist_new
            ml_closest = i
    return ml_closest

def get_center(img, part=.25):
    factor = 32
    y_half, x_half, _ = [i//2 for i in np.shape(img)]
    y_include, x_include = np.asarray([y_half, x_half])* part
    y_include = y_include + (factor - y_include % factor)
    x_include = x_include + (factor - x_include % factor)
    y_part1, x_part1 = int(y_half - y_include), int(x_half - x_include)
    y_part2, x_part2 = int(y_half + y_include), int(x_half + x_include)
    y_part1, y_part2, x_part1, x_part2
    return img[y_part1: y_part2, x_part1: x_part2,:], y_part1, x_part1

def denormalizePatches(img):
    img = img * img_max + img_min
    img = img.astype('int16')
    return img

filename = args.filename
img = plt.imread(f'images_jpg2/{filename}.jpg')
img, _, _ = get_center(img)
img2 = rgb2gray(img)
masks_lesions = otsu_thresh_4largest_component(img2)
mask_lesion = component_closest_center(img2, masks_lesions)
mask_inpain = mask_lesion
mask_skin = np.ones_like(mask_inpain)

# reshape
img = np.swapaxes(img,1,2)
img = np.swapaxes(img,0,1)
#img, mask_inpain, mask_lesion = pad_to_multiple_of_32(img, mask_inpain, mask_lesion)
mask_inpain = mask_inpain.astype('int')
mask_lesion = mask_lesion.astype('int')
mask_skin = mask_skin.astype('int')


# normalize
img_min = np.min(img)
img_max = np.max(img)
img = (img - img_min) / (img_max - img_min)
print(f'img: {np.shape(img), np.shape(mask_inpain), np.shape(mask_lesion), img_min, img_max}')
## MELANOMA


torch.cuda.set_device(args.cuda_gpu)
torch.cuda.empty_cache()

def closure():
    global i
    images_all = []
    
    if param_noise:
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            n = n + n.detach().clone().normal_() * n.std() / 50
    
    net_input = net_input_saved
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        
    out = net(net_input)
    #print(np.shape(net_input), np.shape(out), np.shape(mask_var), np.shape(img_var))
    total_loss = mse(out * mask_var, img_var * mask_var)
    total_loss.backward()
        
    print ('Iteration %05d    Loss %.12f' % (i, total_loss.item()), '\r', end='')
    #if  PLOT and i % show_every == 0:
    if  PLOT:
        out_np = torch_to_np(out)
        image_to_save = out_np
        #if np.shape(out_np)[0] == 1:
            #image_to_save = out_np[0]
        #plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1) # DEL original fun
        #plot_for_gif(image_to_save, num_iter, i) # DEL save images to make gif
        images_all.append(image_to_save)
        
    i += 1    
#     if  PLOT and i % show_every == 0: image_generated = image_to_save
#     else: image_generated = []
    
    return total_loss, images_all

# parser = argparse.ArgumentParser()
# parser.add_argument('skip_idx', type=int, help='skip indices already processed')
# parser.add_argument('cuda_gpu', type=int, help='gpu to use')
# args = parser.parse_args()

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
PLOT = True
imsize = -1
dim_div_by = 64

torch.cuda.empty_cache()

dtype = torch.cuda.FloatTensor

path_img_dest = '/data/OMM/project_results/May_29_2020_melanoma/'
name = args.channels_input_and_layers

# NET_TYPE = 'skip_depth6' # one of skip_depth4|skip_depth2|UNET|ResNet
pad = 'zero' # 'zero' OMM it was reflection
OPT_OVER = 'net'
OPTIMIZER = 'adam'
INPUT = 'noise'
input_depth = channels_input #96#*2 # 
#LR = 0.000001 
num_iter = run_epochs # 3001 # 10001
param_noise = True
show_every = 500
#figsize = 5
reg_noise_std = 0.3 #0.3
mse_error_all = []

torch.cuda.empty_cache()
print(filename)
if np.shape(img)==(0,0,0): 
    print(f'{block_name} skip empty blocks!!!!!!!!!!!!')
    #continue
# Here we dont add batch channels
img_np = img
img_mask_np = mask_inpain
#     img_mask_np = block_mask

# LR FOUND
LR = 0.0002

# INPAINTING
restart_i = 0
restart = True

while restart == True:
    start = time.time()
    print(f'training initialization {restart_i} with LR = {LR:.12f}')
    restart_i += 1

    #lungs_slice, mask_slice, nodule, outside_lungs = read_slices(new_name)
    #img_np, img_mask_np, outside_lungs = make_images_right_size(lungs_slice, mask_slice, nodule, outside_lungs)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)
    img_var = np_to_torch(img_np).type(dtype)
    mask_var = np_to_torch(img_mask_np).type(dtype)

    net = skip(input_depth, img_np.shape[0], 
            num_channels_down = channels_layers, # [64,128,256],#[128]*3,
            num_channels_up   = channels_layers, # [64,128,256],#[128]*3, 
            num_channels_skip = channels_skip, #channels_layers, # [64,128,256],#128]*3, 
            upsample_mode='nearest', filter_skip_size=1, filter_size_up=3, filter_size_down=3, 
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
    net = net.type(dtype)        
    net_input = get_noise2(input_depth, INPUT, img_np.shape[1:], noise_type='n').type(dtype)
    #print(f'net_input: {type(net_input.detach().cpu().numpy)}')
    
    #path_trained_model = f'{path_img_dest}models/v6_Unet_init_sample_{idx}.pt'
    #torch.save(net.state_dict(), path_trained_model)

    #mse_error = []
    i = 0
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    print(f'noise: {type(noise.detach().cpu().numpy())}, {np.shape(noise.detach().cpu().numpy())}, {np.min(noise.detach().cpu().numpy()), np.mean(noise.detach().cpu().numpy()), np.max(noise.detach().cpu().numpy())}')
    #assert 0==1
    p = get_params(OPT_OVER, net, net_input)
    mse_error, images_generated_all, best_iter, restart = optimize_melanoma_v1(OPTIMIZER, p, closure, LR, num_iter, show_every, path_img_dest, restart, annealing=True, lr_finder_flag=False)
    #mse_error = mse_error[0].detach().cpu().numpy()
    
    
    mse_error = [i.detach().cpu().numpy() for i in mse_error]
    #mse_error_all.append(mse_error)
#     mse_error_last = mse_error[-1].detach().cpu().numpy()

    if restart_i % 10 == 0: # reduce lr if the network is not learning with the initializations
        LR /= 1.2
    if restart_i == 30: # if the network cannot be trained continue (might not act on for loop!!)
        continue

print(f'images_generated_all: {np.shape(images_generated_all)}')

#image_last = images_generated_all[-1] * mask_skin
image_orig = img_np[0] * mask_skin
image2000 = images_generated_all[200] * mask_skin
image1600 = images_generated_all[160] * mask_skin
image1200 = images_generated_all[120] * mask_skin
image800 = images_generated_all[80] * mask_skin
image400 = images_generated_all[40] * mask_skin
image200 = images_generated_all[20] * mask_skin


best_iter = f'{best_iter:4d}'

# # convert into ints to occupy less space
#image_last = denormalizePatches(image_last)
image_np = denormalizePatches(img_np)
image2000 = denormalizePatches(image2000)
image1600 = denormalizePatches(image1600) 
image1200 = denormalizePatches(image1200) 
image800 = denormalizePatches(image800) 
image400 = denormalizePatches(image400) 
image200 = denormalizePatches(image200) 

#image_np = img_np * img_max + img_min
#image_np  = image_np.astype('int16')


stop = time.time()
print(f'{(stop-start)/60:.2f} min')
#image_last.tofile(f'{path_img_dest}arrays/last/{name}_{filename}.raw')
image2000.tofile(f'{path_img_dest}arrays/last/{name}_2000_{filename}.raw')
image1600.tofile(f'{path_img_dest}arrays/last/{name}_1600_{filename}.raw')
image1200.tofile(f'{path_img_dest}arrays/last/{name}_1200_{filename}.raw')
image800.tofile(f'{path_img_dest}arrays/last/{name}_0800_{filename}.raw')
image400.tofile(f'{path_img_dest}arrays/last/{name}_0400_{filename}.raw')
image200.tofile(f'{path_img_dest}arrays/last/{name}_0200_{filename}.raw')

image_np.tofile(f'{path_img_dest}arrays/orig/{name}_{filename}.raw')
np.savez_compressed(f'{path_img_dest}arrays/masks_inpain/{name}_{filename}',mask_inpain)
np.savez_compressed(f'{path_img_dest}arrays/masks_lesion/{name}_{filename}',mask_lesion)
np.savez_compressed(f'{path_img_dest}arrays/masks_skin/{name}_{filename}',mask_skin)
np.save(f'{path_img_dest}mse_error_curves_inpainting/{name}_{filename}.npy',mse_error)
#np.save(f'{path_img_dest}box_coords/{name}_{filename}.npy', box_coord)
#torch.save({'epoch': len(mse_error), 'model_state_dict': net.state_dict(),
#    'LR': LR,'loss': mse_error, 'net_input_saved': net_input_saved}, 
#    f'{path_img_dest}v17v2_merged_clusters/models/{name}_{block_name}.pt')
del net, images_generated_all
