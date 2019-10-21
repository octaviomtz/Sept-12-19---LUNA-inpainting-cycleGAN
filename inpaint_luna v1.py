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
# import imageio
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
warnings.filterwarnings("ignore", category=UserWarning)

import SimpleITK as sitk

parser = argparse.ArgumentParser()
parser.add_argument('skip_idx', type=int, help='skip indices already processed')
parser.add_argument('cuda_gpu', type=int, help='gpu to use')
args = parser.parse_args()

torch.cuda.set_device(args.cuda_gpu)
torch.cuda.empty_cache()

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing
def resample_scan_sitk(image,spacing, original_shape, new_spacing=[1,1,1], resampling_method=sitk.sitkLinear):

    # reorder sizes as sitk expects them
    spacing_sitk = [spacing[1],spacing[2],spacing[0]]
    new_spacing_sitk = [new_spacing[1],new_spacing[2],new_spacing[0]]
    
    # set up the input image as at SITK image
    img = sitk.GetImageFromArray(image)
    img.SetSpacing(spacing_sitk)                
            
    # set up an identity transform to apply
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(np.eye(3,3).ravel())
    affine.SetCenter(img.GetOrigin())
    
    # make the reference image grid, original_shape, with new spacing
    refImg = sitk.GetImageFromArray(np.zeros(original_shape,dtype=image.dtype))
    refImg.SetSpacing(new_spacing_sitk)
    refImg.SetOrigin(img.GetOrigin())
    
    imgNew = sitk.Resample(img, refImg, affine ,resampling_method, 0)
    
    imOut = sitk.GetArrayFromImage(imgNew).copy()
    
    return imOut

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

path_seg =  f'/data/OMM/Datasets/LUNA/seg-lungs-LUNA16/'
path_data = f'/data/OMM/Datasets/LUNA/candidates/'
path_img_dest = '/data/OMM/project results/Feb 5 19 - Deep image prior/dip luna v2/'

# NET_TYPE = 'skip_depth6' # one of skip_depth4|skip_depth2|UNET|ResNet
pad = 'zero' # 'zero' OMM it was reflection
OPT_OVER = 'net'
OPTIMIZER = 'adam'
INPUT = 'noise'
input_depth = 96*2 # 
#LR = 0.000001 
num_iter = 5001 # 10001
param_noise = True
show_every = 500
#figsize = 5
reg_noise_std = 0.3

# Create dataset of resampled nodules for classification original and inpainted.ipynb
ids = os.listdir(path_data)
ids = np.sort(ids)

def make3d_from_sparse_v2(path):
    slices_all = os.listdir(path)
    slices_all = np.sort(slices_all)
    for idx, i in enumerate(slices_all):
#         print(idx, i)
        sparse_matrix = sparse.load_npz(f'{path}{i}')
        array2d = np.asarray(sparse_matrix.todense())
        if idx == 0: 
            scan3d = array2d
            continue
        scan3d = np.dstack([scan3d,array2d])
    scan3d = np.swapaxes(scan3d,2,1)
    scan3d = np.swapaxes(scan3d,1,0)
    return scan3d

def median_center(segmentation, label = 1):
    z,y,x = np.where(segmentation == label)
    zz = int(np.median(z))
    yy = int(np.median(y))
    xx = int(np.median(x))
    return zz,yy,xx

def read_slices3D_v4(path_data_, path_seg_, ii_ids):
    """Read VOLUMES of lung, mask outside lungs and nodule, mask nodule, mask outside"""
    #ii_ids = f'LIDC-IDRI-{idnumber:04d}'
    print(f'reading scan {ii_ids}')
    vol = np.load(f'{path_data_}{ii_ids}/lungs_segmented/lungs_segmented.npz')
    vol = vol.f.arr_0
    mask = make3d_from_sparse(f'{path_data_}{ii_ids}/consensus_masks/')
    mask_maxvol = make3d_from_sparse(f'{path_data_}{ii_ids}/maxvol_masks/')
    # read segmentations from luna
    numpyImage, numpyOrigin, numpySpacing = load_itk_image(f'{path_seg_}{ii_ids}.mhd')
    new_spacing = [1,1,1]
    numpyImage_shape = ((np.shape(numpyImage) * numpySpacing) / np.asarray(new_spacing)).astype(int)
    mask_lungs = resample_scan_sitk(numpyImage, numpySpacing, numpyImage_shape, new_spacing=new_spacing)
    np.shape(mask_lungs)
    ##rearrange axes to slices first
    # vol = np.swapaxes(vol,1,2)
    # vol = np.swapaxes(vol,0,1)
    mask = np.swapaxes(mask,1,2)
    mask = np.swapaxes(mask,0,1)
    mask_maxvol = np.swapaxes(mask_maxvol,1,2)
    mask_maxvol = np.swapaxes(mask_maxvol,0,1)
    mask_consensus = mask
    # use only two labels for mask
    mask_lungs = (mask_lungs>0).astype(int)
    return vol, mask_maxvol, mask_consensus, mask_lungs

def small_versions(vol_, mask_maxvol_, mask_consensus_, mask_lungs_):
    z,y,x = np.where(mask_lungs==1)
    z_min = np.min(z); z_max = np.max(z)
    y_min = np.min(y); y_max = np.max(y)
    x_min = np.min(x); x_max = np.max(x)
    vol_small = vol_[z_min:z_max, y_min:y_max, x_min:x_max]
    mask_maxvol_small = mask_maxvol_[z_min:z_max, y_min:y_max, x_min:x_max]
    mask_consensus_small = mask_consensus_[z_min:z_max, y_min:y_max, x_min:x_max]
    mask_lungs_small = mask_lungs_[z_min:z_max, y_min:y_max, x_min:x_max]
    small_boundaries = [z_min, z_max, y_min, y_max, x_min, x_max]
    return vol_small, mask_maxvol_small, mask_consensus_small, mask_lungs_small, small_boundaries

def denormalizePatches(npzarray):
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray * (maxHU - minHU)) + minHU
    npzarray = (npzarray).astype('int16')
    return npzarray

for idx_name, name in enumerate(ids):

    if idx_name < args.skip_idx: continue
    print(idx_name)

    # name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192'
    vol, mask_maxvol, mask_consensus, mask_lungs = read_slices3D_v4(path_data, path_seg, name)
    maxvol0 = np.where(mask_maxvol==1)
    mask_maxvol_and_lungs = deepcopy(mask_lungs)
    mask_maxvol_and_lungs[maxvol0] = 0
    vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small, small_boundaries = small_versions(vol, mask_maxvol, mask_maxvol_and_lungs, mask_lungs)
    vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small = pad_if_vol_too_small(vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small)
    slice_middle = np.shape(vol_small)[0] // 2
    labeled, n_items = ndimage.label(mask_maxvol_small)
    xmed_1, ymed_1, xmed_2, ymed_2 = erode_and_split_mask(mask_lungs_small,slice_middle)
    coord_min_side1, coord_max_side1, coord_min_side2, coord_max_side2 = nodule_right_or_left_lung(mask_maxvol_small, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2)
    block1_list, block1_mask_list, block1_mask_maxvol_and_lungs_list, block1_mask_lungs_list, clus_names1, box_coords1 = get_box_coords_per_block(coord_min_side1, coord_max_side1, 1, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2, vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small, normalize=False)
    block2_list, block2_mask_list, block2_mask_maxvol_and_lungs_list, block2_mask_lungs_list, clus_names2, box_coords2 = get_box_coords_per_block(coord_min_side2, coord_max_side2, 2, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2, vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small, normalize=False)
    blocks_ndl, blocks_ndl_mask, block_mask_maxvol_and_lungs, blocks_ndl_lungs, blocks_ndl_names, box_coords = get_block_if_ndl_list(block1_list, block2_list, block1_mask_list, block2_mask_list, block1_mask_maxvol_and_lungs_list, block2_mask_maxvol_and_lungs_list, block1_mask_lungs_list, block2_mask_lungs_list, clus_names1, clus_names2, box_coords1, box_coords2)
    del vol_small, mask_maxvol_small, mask_consensus, mask_lungs_small
    for (block, block_mask, block_maxvol_and_lungs, block_lungs, block_name, box_coord) in zip(blocks_ndl, blocks_ndl_mask, block_mask_maxvol_and_lungs, blocks_ndl_lungs, blocks_ndl_names, box_coords): 
        torch.cuda.empty_cache()
        print(block_name)
        if np.shape(block)==(0,0,0): 
            #skip empty blocks: (47) 1.3.6.1.4.1.14519.5.2.1.6279.6001.117040183261056772902616195387
            print(f'{block_name} skip empty blocks!!!!!!!!!!!!')
            continue
        # Here we dont add batch channels
        img_np = block
        img_mask_np = block_maxvol_and_lungs
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
                    num_channels_down = [128]*5,
                    num_channels_up   = [128]*5, 
                    num_channels_skip = [128]*5, 
                    upsample_mode='nearest', filter_skip_size=1, filter_size_up=3, filter_size_down=3, 
                    need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
            net = net.type(dtype)        
            net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)

            #path_trained_model = f'{path_img_dest}models/v6_Unet_init_sample_{idx}.pt'
            #torch.save(net.state_dict(), path_trained_model)

            #mse_error = []
            i = 0
            net_input_saved = net_input.detach().clone()
            noise = net_input.detach().clone()
            p = get_params(OPT_OVER, net, net_input)
            mse_error, images_generated_all, best_iter, restart = optimize_v18(OPTIMIZER, p, closure, LR, num_iter, show_every, path_img_dest, restart, annealing=True, lr_finder_flag=False)
            mse_error = np.squeeze(mse_error)
            mse_error = [i.detach().cpu().numpy() for i in mse_error]
        #     mse_error_all.append(mse_error)
        #     mse_error_last = mse_error[-1].detach().cpu().numpy()

            if restart_i % 10 == 0: # reduce lr if the network is not learning with the initializations
                LR /= 1.2
            if restart_i == 30: # if the network cannot be trained continue (might not act on for loop!!)
                continue
        print('')
        #print(np.shape(images_generated_all))
        print('')
        image_last = images_generated_all[-1] * block_lungs
        image_orig = img_np[0] * block_lungs
        best_iter = f'{best_iter:4d}'

        # convert into ints to occupy less space
        image_last = denormalizePatches(image_last)
        img_np = denormalizePatches(img_np)

        stop = time.time()
        image_last.tofile(f'{path_img_dest}arrays/last/{name}_{block_name}.raw')
        img_np.tofile(f'{path_img_dest}arrays/orig/{name}_{block_name}.raw')
        np.savez_compressed(f'{path_img_dest}arrays/masks/{name}_{block_name}',block_maxvol_and_lungs)
        np.savez_compressed(f'{path_img_dest}arrays/masks nodules/{name}_{block_name}',block_mask)
        np.savez_compressed(f'{path_img_dest}arrays/masks lungs/{name}_{block_name}',block_lungs)
        np.save(f'{path_img_dest}mse error curves inpainting/{name}_{block_name}.npy',mse_error)
        np.save(f'{path_img_dest}inpainting times/{name}_{block_name}_{int(stop-start)}s.npy',stop-start)
        np.save(f'{path_img_dest}box_coords/{name}_{block_name}.npy', box_coord)
        #torch.save({'epoch': len(mse_error), 'model_state_dict': net.state_dict(),
        #    'LR': LR,'loss': mse_error, 'net_input_saved': net_input_saved}, 
        #    f'{path_img_dest}v17v2_merged_clusters/models/{name}_{block_name}.pt')
        del net, images_generated_all
