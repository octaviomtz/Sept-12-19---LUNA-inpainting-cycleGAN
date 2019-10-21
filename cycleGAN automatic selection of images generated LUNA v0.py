'''insert GAN images for object detection'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import copy
from matplotlib import rcParams
import matplotlib.patches as patches
from scipy.ndimage import label
# from tqdm import tqdm_notebook
from scipy.stats import ks_2samp
from scipy import ndimage
from scipy import sparse
import SimpleITK as sitk
import pickle
import sys
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Functions

def LDS( A ):
    '''Largest Decreasing Sequence'''
    # http://www.geekviewpoint.com/python/dynamic_programming/lds
    m = [0] * len( A ) # starting with m = [1] * len( A ) is not necessary
    for x in range( len( A ) - 2, -1, -1 ):
        for y in range( len( A ) - 1, x, -1 ):
            if m[x] <= m[y] and A[x] > A[y]:
                m[x] = m[y] + 1 # or use m[x]+=1
 
    #===================================================================
    # Use the following snippet or the one line below to get max_value
    # max_value=m[0]
    # for i in range(m):
    #  if max_value < m[i]:
    #    max_value = m[i]
    #===================================================================
    max_value = max( m )
 
    result = []
    for i in range( len( m ) ):
        if max_value == m[i]:
            result.append( A[i] )
            max_value -= 1
 
    return result


def takespread(sequence, num):
    '''
    choose-m-evenly-spaced-elements-from-a-sequence-of-length-n
    https://stackoverflow.com/questions/9873626/choose-m-evenly-spaced-elements-from-a-sequence-of-length-n
    '''
    length = float(len(sequence))
    for i in range(num):
        yield sequence[int(np.ceil(i * length / num))]



path_source = '/data/OMM/project results/Feb 20 19 - CycleGan clean/deep nodule prior luna v2 - cubes size 32 coefficients/A/' # GAN generated images
path_source_original = '/data/OMM/Datasets/LIDC_other_formats/LUNA_inpainted_cubes_for_GAN_v2/' # inpainted images (input for GAN)
# path_source_original = '/data/OMM/Datasets/LIDC_other_formats/LUNA_inpainted_cubes_for_GAN/' # inpainted images (input for GAN)
path_orig = f'{path_source_original}original/'
path_inpain = f'{path_source_original}inpainted inserted/'
path_mask = f'{path_source_original}mask/'
path_dest = f'{path_source[:-2]}images augmented/'
data_candidates  = '/data/OMM/Datasets/LUNA/'
files_all = os.listdir(path_source)
files_all = np.sort(files_all)
files_unique = [i.split('_ep')[0] for i in files_all]
files_unique = np.unique(files_unique)
files_unique = np.sort(files_unique)
len(files_unique)



def normalizePatches(npzarray):
    maxHU = 400.
    minHU = -1000.

    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

def get_files_ndl_centered(path_, file_):
    origA = np.fromfile(f'{path_}original/{file_}',dtype='int16').astype('float32').reshape((64,64,64))
    origB = np.fromfile(f'{path_}inpainted inserted/{file_}',dtype='int16').astype('float32').reshape((64,64,64))
    origA = normalizePatches(origA)
    origB = normalizePatches(origB)
    mask = np.fromfile(f'{path_}mask/{file_}',dtype='int16').reshape((64,64,64))
    origA = origA[16:-16,16:-16,16:-16]
    origB = origB[16:-16,16:-16,16:-16]
    mask = mask[16:-16,16:-16,16:-16]
    return origA, origB, mask


def get_images_similar_to_original(files_all_, file_, path_, origA_, mask_ndl_):
    '''Get all the images similar to the original'''
    images_gen = []
    files_one_nodule = [i for i in files_all_ if file_ in i]
    idx_true = []
    #print(f'epochs saved: {len(files_one_nodule)}')
    for idx, i in enumerate(files_one_nodule):
        a = np.fromfile(f'{path_}{i}',dtype='float32').reshape((32,32,32))
        mean_diff = np.mean(np.abs(origA_*(-mask_ndl_+1) - a*(-mask_ndl_+1)))

        if mean_diff < 0.04: # original 0.03
            images_gen.append(a)
            idx_true.append(idx)
    return images_gen, idx_true


def image_of_growing_nodules_together(images_genA_all_, lds_idxs_, origB_, origA_, mask_correct_, file_one_):
    imgs_ndl_area = []
    np.shape(images_genA_all_)
    num_rows = int(len(lds_idxs_)// 5 +1 )
    fig, ax = plt.subplots(num_rows,5, figsize=(14,int(3*num_rows)))
    idxx = 0 # for idx of images
    for idx, i in enumerate(images_genA_all_):
        if idx == (len(images_genA_all_)//5)*5: break
        if idx in lds_idxs_:
            # generated over inpainted
            image_patched = i*mask_correct_ + (origB_*(-mask_correct_+1))
            # generated over inpainted only relevant areas (intensities higher than .1)
            image_patched_relevant = i*(mask_correct_ & np.abs((i-origB_)>.1)) + (origB_*(-(mask_correct_ & np.abs((i-origB_)>.1))+1))
            ax[idxx//5,np.mod(idxx,5)].imshow(image_patched_relevant, vmin=0, vmax=1)
            ax[idxx//5,np.mod(idxx,5)].set_title(idx_true[idx]*2)
            imgs_ndl_area.append(image_patched)
            idxx+=1
    ax[idxx//5,np.mod(idxx,5)].imshow(origA_, vmin=0, vmax=1)
    ax[idxx//5,np.mod(idxx,5)].set_title('orig', color='r')
    for axx in ax.ravel(): axx.axis('off')
    fig.tight_layout()
    fig.savefig(f'{path_dest}images growing nodules/{file_one_[:-4]}.jpg')
    plt.close()

def image_of_growing_nodules_together2(images_genA_all_, lds_idxs_, origB_, origA_, mask_correct_, file_one_):
    imgs_ndl_area = []
    np.shape(images_genA_all_)
    num_rows = int(len(lds_idxs_)// 5 +1 )
    fig, ax = plt.subplots(num_rows,5, figsize=(14,int(3*num_rows)))
    idxx = 0 # for idx of images
    for idx, i in enumerate(images_genA_all_):
        ax[idx//5,np.mod(idx,5)].imshow(i, vmin=0, vmax=1)
        ax[idx//5,np.mod(idx,5)].set_title(idx_true[idx]*2)
        #imgs_ndl_area.append(image_patched)
        idxx+=1
    ax[idx//5,np.mod(idx,5)].imshow(origA_, vmin=0, vmax=1)
    ax[idx//5,np.mod(idx,5)].set_title('orig', color='r')
    for axx in ax.ravel(): axx.axis('off')
    fig.tight_layout()
    fig.savefig(f'{path_dest}images growing nodules/{file_one_[:-4]}.jpg')
    plt.close()

def image_of_growing_nodules_together3(images_genA_all_, lds_idxs_, origB_, origA_, mask_correct_, file_one_):
    '''save images of growing nodules according to LDS in an image collage '''
    imgs_ndl_area = []
    np.shape(images_genA_all_)
    num_rows = int(len(lds_idxs_)// 5 +1 )
    fig, ax = plt.subplots(num_rows,5, figsize=(14,int(3*num_rows)))
    idxx = 0 # for idx of images
    for idx, i in enumerate(images_genA_all_):
        ax[idx//5,np.mod(idx,5)].imshow(i, vmin=0, vmax=1)
        ax[idx//5,np.mod(idx,5)].set_title(idx_true[idx]*2)
        #imgs_ndl_area.append(image_patched)
        idxx+=1
    ax[idx//5,np.mod(idx,5)].imshow(origA_, vmin=0, vmax=1)
    ax[idx//5,np.mod(idx,5)].set_title('orig', color='r')
    for axx in ax.ravel(): axx.axis('off')
    fig.tight_layout()
    if os.path.isdir(f'{path_dest}images growing nodules/') == False:
        os.makedirs(f'{path_dest}images growing nodules/') 
    fig.savefig(f'{path_dest}images growing nodules/{file_one_[:-4]}.jpg')
    plt.close()

def median_center(segmentation, label = 1):
    z,y,x = np.where(segmentation == label)
    zz = int(np.median(z))
    yy = int(np.median(y))
    xx = int(np.median(x))
    return zz,yy,xx

def find_indices_of_LDS(lds_values, xymedian_diff_all):
    '''Find the indices of the longest decreasing sequence
    np.where returns all the indices where a value was found. 
    We find the idx that follows our current (saved) idx'''

    lds_idxs = []
    idx_previous = -1
    flag_continue = False 
    for i in lds_values:
        idxs_options = np.where(xymedian_diff_all == i)[0] 
        try:
            idxs_options_idx = next(idx for idx, val in enumerate(idxs_options) if val > idx_previous)
        except StopIteration: 
            print('sampled skipped, StopIteration')
            flag_continue = True
        idx_previous = idxs_options[idxs_options_idx]
        lds_idxs.append(idx_previous)

    if len(lds_idxs) <=2:
        print('sampled skipped, too few idxs')
        flag_continue = True
    return lds_idxs, flag_continue


def make_patch_generated_dirs(path_dest, file_one):
    '''Make the folders to save the images from the LDS in a 
    corresping folder for each nodule'''
#     if os.path.isdir(f'{path_dest}generated/{file_one[:-4]}') == False:
#         os.makedirs(f'{path_dest}generated/{file_one[:-4]}')
#     if os.path.isdir(f'{path_dest}patched/{file_one[:-4]}') == False:
#         os.makedirs(f'{path_dest}patched/{file_one[:-4]}')
#     if os.path.isdir(f'{path_dest}images growing nodules/') == False:
#         os.makedirs(f'{path_dest}images growing nodules/') 

def blend_nodule_growth_gan_inpainted_and_original(origA, origB, i, mask_correct):
    '''mask_gan: abs(i-origB) shows the growth of nodule against the inpainted image. We only take
    large differences (>.1) and we only take the differences inside the nodule
    image_patched_relevant: formed from three images (with three masks). the nodule growing (mask_gan)
    the inpainted image (to fill the area around the nodule growing) and the original image (to fill the area around the complete nodule mask)'''
    #previous
    #image_patched_relevant = i*(mask_correct & np.abs((i-origB)>.1)) + (origA*(-(mask_correct & np.abs((i-origB)>.1))+1))
    mask_gan = mask_correct & np.abs((i-origB)>.1)
    image_patched_relevant = (i * mask_gan) + (mask_correct * origB)*(-mask_gan+1) +  ((-mask_correct+1) * origA)
    return image_patched_relevant

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


def denormalizePatches(npzarray):
    
    maxHU = 400.
    minHU = -1000.
 
    npzarray = (npzarray * (maxHU - minHU)) + minHU
    npzarray = (npzarray).astype('int16')
    return npzarray



def find_no_segmented_subfolder_and_read_it(file_one, data_candidates):
    '''find the corresponding subfolder of this image, read it and resample it'''
    # find the corresponding subfolder of this image
    name_original = file_one.split('_')[0] + '.mhd'
    data_dir_subsets = os.listdir(data_candidates)
    data_dir_subsets = [i for i in data_dir_subsets if 'subset' in i]
    data_dir_subsets = np.sort(data_dir_subsets)
    for sub in data_dir_subsets:
        if name_original in os.listdir(f'{data_candidates}{sub}'):
            break
    #read original (no-segmented) image
    numpyImage, numpyOrigin, numpySpacing = load_itk_image(f'{data_candidates}{sub}/{name_original}')
    # resample the image_without_segmentation
    new_spacing = [1,1,1]
    numpyImage_shape = ((np.shape(numpyImage) * numpySpacing) / np.asarray(new_spacing)).astype(int)
    numpyImage_resampled = resample_scan_sitk(numpyImage, numpySpacing, numpyImage_shape, new_spacing=new_spacing)
    return numpyImage_resampled, sub


def insert_gan_image_in_no_segmented_lungs(image_patched_relevant, file_one, numpyImage_resampled):
    '''1. the coords of the nodule are in its name.
    2. get a copy of the block around the nodule
    3. denormalize gan-created image and make sure its () similar to the resampled block 
    4. insert gan-created image (just the nodule mask) into the block
    5. put back the block (with the inserted gan-created nodule) into the copy of the resampled image'''
    size_half = 16
    # the coords of the nodule are in its name
    coord_z = int(file_one.split('_z')[-1].split('y')[0])
    coord_y = int(file_one.split('y')[-1].split('x')[0])
    coord_x = int(file_one.split('x')[-1].split('.')[0])
    # get a copy of the block around the nodule
    numpyImage_resampled_inserted = copy(numpyImage_resampled)
    numpyImage_resampled_block = numpyImage_resampled_inserted[coord_z-size_half:coord_z+size_half, coord_y-size_half:coord_y+size_half, coord_x-size_half:coord_x+size_half]
    np.shape(numpyImage_resampled_block)
    # denormalize gan-created image and make sure its () similar to the resampled block 
    image_patched_denorm = denormalizePatches(image_patched_relevant)
    assert np.sum(numpyImage_resampled_block == image_patched_denorm) > 1000
    # insert gan-created image (just the nodule mask) into the block
    zz,yy,xx = np.where(mask_correct==1)
    numpyImage_resampled_block[zz,yy,xx] = image_patched_denorm[zz,yy,xx]
    # put back the block (with the inserted gan-created nodule) into the copy of the resampled image
    numpyImage_resampled_inserted[coord_z-size_half:coord_z+size_half, coord_y-size_half:coord_y+size_half, coord_x-size_half:coord_x+size_half] = numpyImage_resampled_block
    return numpyImage_resampled_inserted, coord_z, coord_y, coord_x

for idx_unique, file_one in enumerate(files_unique):
    if idx_unique <= 1659: continue #skipped: 53. 333, 567, 617, 976, 1658
    origA, origB, mask_correct = get_files_ndl_centered(path_source_original, file_one)
    images_genA_all, idx_true = get_images_similar_to_original(files_all, file_one, path_source, origA, mask_correct)
    z,y,x = np.where(mask_correct==1)
    
    pixels_epochs_all = []
    xymedian_diff_all = []
    for idx, i in enumerate(images_genA_all):
        pixels_epochs = i[z,y,x]
        pixels_orig = origA[z,y,x]
        pixels_inpain = origB[z,y,x]
        xymedian_diff = np.median(pixels_orig-pixels_epochs)
        pixels_epochs_all.append(pixels_epochs)
        xymedian_diff_all.append(xymedian_diff)
    try:
        lds_values = LDS(xymedian_diff_all) 
    except ValueError: 
        print(f'ValueError at {idx_unique}, ({file_one})')
        continue

    lds_idxs, flag_continue = find_indices_of_LDS(lds_values, xymedian_diff_all)
    if flag_continue: continue

    # We only saved every other epoch, we need to plot the correct epoch
    idx_true_every_other = np.asarray(idx_true)*2  

    # make_patch_generated_dirs(path_dest, file_one)

    images_genA_all_patched, images_genA_all_patched2 = [], []
    lds_idxs_save = []

    numpyImage_resampled, subfolder_save = find_no_segmented_subfolder_and_read_it(file_one, data_candidates)

    for idx, (i,j) in enumerate(zip(images_genA_all, idx_true)):    
        image_patched_relevant = blend_nodule_growth_gan_inpainted_and_original(origA, origB, i, mask_correct)
        numpyImage_resampled_inserted, coord_z, coord_y, coord_x = insert_gan_image_in_no_segmented_lungs(image_patched_relevant, file_one, numpyImage_resampled)
        # if idx in lds_idxs: # we used to save the LDS
        cube_half_size = 40
        cube_augmented = numpyImage_resampled_inserted[coord_z-cube_half_size:coord_z+cube_half_size, coord_y-cube_half_size:coord_y+cube_half_size, coord_x-cube_half_size:coord_x+cube_half_size]
        if os.path.isdir(f'{path_dest}patched/{subfolder_save}/{file_one[:-4]}') == False:
            os.makedirs(f'{path_dest}patched/{subfolder_save}/{file_one[:-4]}')
        cube_augmented.tofile(f'{path_dest}patched/{subfolder_save}/{file_one[:-4]}/{file_one[:-4]}_ep{j:03d}')
        if idx_unique < 50:
            images_genA_all_patched.append(image_patched_relevant[15])
            images_genA_all_patched2.append(normalizePatches(cube_augmented[cube_half_size-1]))
            lds_idxs_save.append(idx)
    try:
        if idx_unique < 50:
            image_of_growing_nodules_together3(images_genA_all_patched, lds_idxs_save, origB[15], origA[15], mask_correct, file_one)
            file_one_full = f'full_{file_one}'
            image_of_growing_nodules_together3(images_genA_all_patched2, lds_idxs_save, origB[15], origA[15], mask_correct, file_one_full)
    except IndexError: 
        print('IndexError when creating figure')
        continue
    perc_done = 100 * idx_unique / len(files_unique)
    sys.stdout.write(f'\r{perc_done:.2f}% done ({idx_unique} of {len(files_unique)})')