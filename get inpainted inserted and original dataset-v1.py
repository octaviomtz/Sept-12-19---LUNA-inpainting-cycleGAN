# ## find the nodules in the inpainted images and make cubes around them
# 1. v1 saving dataframes with lidc information


import os
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from scipy import ndimage
from copy import copy, deepcopy
import pandas as pd
import SimpleITK as sitk
import pylidc as pl
get_ipython().run_line_magic('matplotlib', 'inline')

##

def median_center(segmentation, label = 1):
    z,y,x = np.where(segmentation == label)
    zz = int(np.median(z))
    yy = int(np.median(y))
    xx = int(np.median(x))
    return zz,yy,xx

##

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

##

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

##

path_source = f'/data/OMM/Datasets/LUNA/'
path_data = f'/data/OMM/Datasets/LUNA/candidates/'
path_inpainting_done = '/data/OMM/project results/Feb 5 19 - Deep image prior/dip luna v1/arrays/' # previosu
path_inpainting_done = '/data/OMM/project results/Feb 5 19 - Deep image prior/dip luna v2/arrays/' # new
path_last = f'{path_inpainting_done}last/'
path_orig = f'{path_inpainting_done}orig/'
path_masks = f'{path_inpainting_done}masks/'
path_mask_nodules = f'{path_inpainting_done}masks nodules/'
path_boxes_coords = f'{path_inpainting_done[:-7]}box_coords/'
path_dest = '/data/OMM/Datasets/LIDC_other_formats/LUNA_inpainted_cubes_for_GAN_v1/' # previous
path_dest = '/data/OMM/Datasets/LIDC_other_formats/LUNA_inpainted_cubes_for_GAN_v2/' # new
path_qualitative_evaluation = f'{path_dest}versions2D/qualitative assessment/'
path_seg = '/data/OMM/Datasets/LUNA/seg-lungs-LUNA16/'
ff = os.listdir(path_last)
ff = np.sort(ff)

##

def load_inpainted_images(file_name_):
    '''load the inpainting results. These are obtained from the blocks (~[96,160,96])'''
    last = np.load(f'{path_last}{file_name_}')
    last = np.squeeze(last)
    orig = np.load(f'{path_orig}{file_name_}')
    masks = np.load(f'{path_masks}{file_name_[:-1]}z')
    masks = masks.f.arr_0
    mask_nodules = np.load(f'{path_mask_nodules}{file_name_[:-1]}z')
    mask_nodules = mask_nodules.f.arr_0
    inserted = mask_nodules*last + (-mask_nodules+1)*orig
    # find nodule centers
    labeled, nr = ndimage.label(mask_nodules)
    ndls_centers = []
    for i in np.arange(1, nr+1):
        zz, yy, xx = median_center(labeled, i)
        ndls_centers.append([zz,yy,xx])
    return last, orig, masks, mask_nodules, inserted, ndls_centers

##

def load_inpainted_images(file_name_):
    '''load the inpainting results. These are obtained from the blocks (~[96,160,96])'''
    last = np.fromfile(f'{path_last}{file_name_}',dtype='int16').astype('float32').reshape((96,160,96))
    #last = np.squeeze(last)
    orig = np.fromfile(f'{path_orig}{file_name_}',dtype='int16').astype('float32').reshape((96,160,96))
    masks = np.load(f'{path_masks}{file_name_[:-3]}npz')
    masks = masks.f.arr_0
    mask_nodules = np.load(f'{path_mask_nodules}{file_name_[:-3]}npz')
    mask_nodules = mask_nodules.f.arr_0
    inserted = mask_nodules*last + (-mask_nodules+1)*orig
    # find nodule centers
    labeled, nr = ndimage.label(mask_nodules)
    ndls_centers = []
    for i in np.arange(1, nr+1):
        zz, yy, xx = median_center(labeled, i)
        ndls_centers.append([zz,yy,xx])
    return last, orig, masks, mask_nodules, inserted, ndls_centers

##

def load_resampled_image(file_name_):
    '''load resampled scan and masks of the nodules '''
    file_name_ = f.split('_')[0]
    lungs = np.load(f'{path_data}{file_name_}/lungs_segmented/lungs_segmented.npz')
    lungs = lungs.f.arr_0
    mask = make3d_from_sparse_v2(f'{path_data}{file_name_}/maxvol_masks/')
    labeled, nr = ndimage.label(mask)
    ndls_centers = []
    for i in np.arange(1, nr+1):
        zz, yy, xx = median_center(labeled, i)
        ndls_centers.append([zz,yy,xx])
    return lungs, ndls_centers, mask

##

def get_smaller_versions(lungs, mask):
    '''Use the mask of the lungs to create the smaller versions like the ones used in preprocessing.
    Return the smaller versions and the coords (min). These are needed to find the nodules on the small versions'''
    z,y,x = np.where(lungs>0)
    x_max = np.max(x); x_min = np.min(x);
    y_max = np.max(y); y_min = np.min(y);
    z_max = np.max(z); z_min = np.min(z);
    lungs_small = lungs[z_min:z_max, y_min:y_max, x_min:x_max]
    mask_small = mask[z_min:z_max, y_min:y_max, x_min:x_max]
    return lungs_small, mask_small, z_min, y_min, x_min

##

def get_the_block_from_the_resampled_image(ndls_centers, z_min, y_min, x_min, lungs_small, mask_small, boxes_coords):
    '''make sure that the block is from the right position. We confirm this by comparing the nodules' masks''' 
    ndls_centers_small = [np.asarray(i)- np.asarray([z_min, y_min, x_min]) for i in ndls_centers]
    block_from_resampled = lungs_small[boxes_coords[0]:boxes_coords[1], boxes_coords[2]:boxes_coords[3], boxes_coords[4]:boxes_coords[5]]
    mask_from_resampled = mask_small[boxes_coords[0]:boxes_coords[1], boxes_coords[2]:boxes_coords[3], boxes_coords[4]:boxes_coords[5]]
    # make sure mask from inpainting results == block from resampled
    # we might need to reshape because mask from inpainting results might be smaller (really close to the border)
    sz1,sy1,sx1 = np.shape(mask_nodules)
    sz2,sy2,sx2 = np.shape(mask_from_resampled)
    sz,sy,sx = np.min([sz1, sz2]), np.min([sy1, sy2]), np.min([sx1, sx2])
    assert (mask_from_resampled[0:sz,0:sy,0:sx] == mask_nodules[0:sz,0:sy,0:sx]).all() 
    return block_from_resampled, mask_from_resampled

##

def put_inpainted_in_resampled_image(lungs, inserted, z_min, y_min, x_min, boxes_coords):
    '''We insert the inpainted nodule (from the inserted block) into the resampled image'''
    z_small_plus_boxfound = z_min + boxes_coords[0]
    y_small_plus_boxfound = y_min + boxes_coords[2]
    x_small_plus_boxfound = x_min + boxes_coords[4]
    z_small_plus_boxfound, y_small_plus_boxfound, x_small_plus_boxfound
    shape_block = np.shape(last)
    lungs_inserted = deepcopy(lungs)
    lungs_inserted[z_small_plus_boxfound:z_small_plus_boxfound+shape_block[0], y_small_plus_boxfound:y_small_plus_boxfound+shape_block[1], x_small_plus_boxfound:x_small_plus_boxfound+shape_block[2]] = inserted
    return lungs_inserted, z_small_plus_boxfound, y_small_plus_boxfound, x_small_plus_boxfound

##

def padd_if_mask_close_to_edge(zz, lungs, cube_half, axis):
    padd_z_low = 0
    padd_z_up = 0
    if (zz - cube_half) < 0:
        padd_z_low = cube_half - zz
    if (zz + cube_half) > np.shape(lungs)[axis]:
        zz = np.shape(lungs)[0]
        padd_z_up = cube_half - (np.shape(lungs)[0] - zz)
    return padd_z_low, padd_z_up

def get_cubes_for_gan(mask_nodules, lungs, lungs_inserted, mask_big):
    '''we get a cube around each nodule of the original and the inserted resampled images.
       To do so, we count the mask_nodules of the block and find their centers. Then we find the corresponding
       coords in the resampled image by adding _small_plus_boxfound to these coords.
       we are suppossed to get a cube of size _cube_size_ but if the coord is too close to the edge we take a 
       smaller portion (until the edge) and pad later with 0s'''
    cube_size = 64
    cube_half = cube_size // 2
    cubes_for_gan_inpain = []
    cubes_for_gan_orig = []
    mask_ndls = []
    zzs,yys,xxs = [], [], []
    labeled, nr = ndimage.label(mask_nodules)
    for i in np.arange(1, nr + 1):
        zz,yy,xx = median_center(labeled, i)
        # find the corresponding coords in the resampled image by adding _small_plus_boxfound to these coords
        zz = zz + z_small_plus_boxfound
        yy = yy + y_small_plus_boxfound
        xx = xx + x_small_plus_boxfound
        # we are suppossed to get a cube of size _cube_size_ but if the coord is too close to the edge we take a 
        # smaller portion and pad later
        padd_z_low, padd_z_up = padd_if_mask_close_to_edge(zz, lungs, cube_half, 0)
        padd_y_low, padd_y_up = padd_if_mask_close_to_edge(yy, lungs, cube_half, 1)
        padd_x_low, padd_x_up = padd_if_mask_close_to_edge(xx, lungs, cube_half, 2)
        # get the cube
        cubes_for_gan_orig_temp = lungs[zz-(cube_half-padd_z_low):zz+(cube_half-padd_z_up),yy-(cube_half-padd_y_low):yy+(cube_half-padd_y_up),xx-(cube_half-padd_x_low):xx+(cube_half-padd_x_up)]
        cubes_for_gan_inpain_temp = lungs_inserted[zz-(cube_half-padd_z_low):zz+(cube_half-padd_z_up),yy-(cube_half-padd_y_low):yy+(cube_half-padd_y_up),xx-(cube_half-padd_x_low):xx+(cube_half-padd_x_up)]
        mask_ndl_from_big_temp = mask_big[zz-(cube_half-padd_z_low):zz+(cube_half-padd_z_up),yy-(cube_half-padd_y_low):yy+(cube_half-padd_y_up),xx-(cube_half-padd_x_low):xx+(cube_half-padd_x_up)]
        
        # pad if needed
        cubes_for_gan_orig_temp = np.pad(cubes_for_gan_orig_temp, ((padd_z_low,padd_z_up), (padd_y_low,padd_y_up), (padd_x_low, padd_x_up)), 'constant', constant_values=0)
        cubes_for_gan_inpain_temp = np.pad(cubes_for_gan_inpain_temp, ((padd_z_low,padd_z_up), (padd_y_low,padd_y_up), (padd_x_low, padd_x_up)), 'constant', constant_values=0)
        mask_ndl_from_big_temp = np.pad(mask_ndl_from_big_temp, ((padd_z_low,padd_z_up), (padd_y_low,padd_y_up), (padd_x_low, padd_x_up)), 'constant', constant_values=0)
        
        # coords of the cube_for_gan with respect of the resampled image
        zz_cube_resampled = zz + padd_z_low - cube_half
        yy_cube_resampled = yy + padd_y_low - cube_half
        xx_cube_resampled = xx + padd_x_low - cube_half
        paddings = [padd_z_low, padd_z_up, padd_y_low, padd_y_up, padd_x_low, padd_x_up]
        
        # Append
        cubes_for_gan_orig.append(cubes_for_gan_orig_temp)
        cubes_for_gan_inpain.append(cubes_for_gan_inpain_temp)
        mask_ndls.append(mask_ndl_from_big_temp)
        zzs.append(zz)
        yys.append(yy)
        xxs.append(xx)
    return cubes_for_gan_orig, cubes_for_gan_inpain, mask_ndls, zzs, yys, xxs, zz_cube_resampled, yy_cube_resampled, xx_cube_resampled, paddings

##

def normalizePatches(npzarray):
    npzarray = npzarray
    
    maxHU = 400.
    minHU = -1000.
 
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

##

def denormalizePatches(npzarray):
    
    maxHU = 400.
    minHU = -1000.
 
    npzarray = (npzarray * (maxHU - minHU)) + minHU
    npzarray = (npzarray).astype('int16')
    return npzarray

##

def qualitative_evaluation_image(orig, inpain, mask, path_save_images, name):
    plt.style.use('dark_background');
    fig, ax = plt.subplots(3,1,figsize=(5,15));
    ax[0].imshow(orig, vmin=0, vmax=1)
    ax[1].imshow(inpain, vmin=0, vmax=1)
    # ax[2].imshow(np.abs(orig-inpain))
    ax[2].imshow(mask)
    for axx in ax.ravel(): axx.axis('off')
    fig.tight_layout()
    fig.savefig(f'{path_save_images}{name[:-4]}.jpg');
    plt.close()
    plt.style.use('default')

##

def qualitative_evaluation_image2(orig, inpain, mask, path_save_images, name):
    plt.style.use('dark_background');
    fig, ax = plt.subplots(3,2,figsize=(9,19.5));
    ax[0,0].imshow(orig, vmin=0, vmax=1)
    ax[0,1].imshow(orig, vmin=0, vmax=1)
    ax[0,1].imshow(mask, alpha = .3)
    ax[1,0].imshow(inpain, vmin=0, vmax=1)
    ax[1,1].imshow(inpain, vmin=0, vmax=1)
    ax[1,1].imshow(mask, alpha = .3)
    ax[2,0].imshow(mask)
    ax[2,1].imshow(np.abs(orig-inpain))
    for axx in ax.ravel(): axx.axis('off')
    fig.tight_layout()
    fig.savefig(f'{path_save_images}{name[:-4]}.jpg');
    plt.close()
    plt.style.use('default')
##

def compare_lidc_coords(name_scan, coords_obtained, numpySpacing):
    '''compare_lidc_coords_against_coords_obtained_in the script and return a DF 
    with the corresponding LIDC information'''
    df_to_return = pd.DataFrame()
    df_annotations_scan = df.loc[df['seriesuid'] == name_scan]
    nodules_unique = np.unique(df_annotations_scan['cluster_id'].values)
    for i_nodules_unique in nodules_unique:
        df_annotations_scan_ndl = df_annotations_scan.loc[df_annotations_scan['cluster_id'] == i_nodules_unique]
        coordZ = int(np.mean(df_annotations_scan_ndl['lidc_coordZ'].values))
        coordY = int(np.mean(df_annotations_scan_ndl['lidc_coordY'].values))
        coordX = int(np.mean(df_annotations_scan_ndl['lidc_coordX'].values))

        coordZ_resampled = int(coordZ * numpySpacing[0])
        coordY_resampled = int(coordY * numpySpacing[1])
        coordX_resampled = int(coordX * numpySpacing[2])
        
        # Compute differences (for some reason we need to compare X-Y)
        diff_Z = coords_obtained[0] - coordZ_resampled       
        diff_Y = coords_obtained[1] - coordX_resampled       
        diff_X = coords_obtained[2] - coordY_resampled
        
        # If the coords are close
        if diff_Z <= 2 and diff_Y <= 2 and diff_X <= 2:
            df_to_return = df_annotations_scan_ndl
            break
    return df_to_return

##

# Added in v2, df from pylidc
df = pd.read_csv('/data/datasets/LIDC-IDRI/annotations.csv')

name_previous = 'no.one'
for idf, f in enumerate(ff):
#     iii=0
#     if idf < iii: continue

    last, orig, masks, mask_nodules, inserted, ndls_centers_block = load_inpainted_images(f)
    inserted = normalizePatches(inserted)
    lungs, ndls_centers, mask = load_resampled_image(f)
    lungs_small, mask_small, z_min, y_min, x_min = get_smaller_versions(lungs, mask)
    boxes_coords = np.load(f'{path_boxes_coords}{f[:-3]}npy') # get the coords of the block (obtained in inpaint_luna.py)
    block_from_resampled, mask_from_resampled = get_the_block_from_the_resampled_image(ndls_centers, z_min, y_min, x_min, lungs_small, mask_small, boxes_coords)
    lungs_inserted, z_small_plus_boxfound, y_small_plus_boxfound, x_small_plus_boxfound = put_inpainted_in_resampled_image(lungs, inserted, z_min, y_min, x_min, boxes_coords)
    cubes_gan_orig, cubes_gan_inpain, mask_ndls, zzs, yys, xxs, zz_cube_resampled, yy_cube_resampled, xx_cube_resampled, paddings = get_cubes_for_gan(mask_nodules, lungs, lungs_inserted, mask)
    
    # the ifs clauses before and after the next *for loop* are to avoid repeating cubes:
    # a nodule might be captured by more than one block but we can identify this by 
    # checking the coordinates with respect to resampled (in file name)
    name_scan = f.split('_')[0]
    _, _, spacing = load_itk_image(f'{path_seg}{name_scan}.mhd') # used in compare_lidc_coords
    if name_scan != name_previous:  
        name_coords_in_scan_all = []
        name_previous = name_scan
    for idx_cube, (cube_orig, cube_inpain, mask_ndl, z,y,x) in enumerate(zip(cubes_gan_orig, cubes_gan_inpain, mask_ndls, zzs, yys, xxs)):
        name_coords_in_scan = f'z{z}y{y}x{x}'
        if name_coords_in_scan in name_coords_in_scan_all: continue # nodule contained in another cube
        else: name_coords_in_scan_all.append(name_coords_in_scan)
        
        coords_obtained = [z,y,x]
        df_lidc_ndl = compare_lidc_coords(name_scan, coords_obtained, spacing)
        
        f_name = f'{f[:-4]}_{name_coords_in_scan}.raw'
        assert np.shape(cube_orig) == (64, 64, 64) and np.shape(cube_inpain) == (64, 64, 64) and np.shape(mask_ndl) == (64, 64, 64)
        # qualitative evaluation
        qualitative_evaluation_image2(cube_orig[31], cube_inpain[31], mask_ndl[31], path_qualitative_evaluation, f_name)
        # denormalize to save in more convenient format 
        cube_orig = denormalizePatches(cube_orig)
        cube_inpain = denormalizePatches(cube_inpain)
        print(idf, f_name)
        # save files
        df_lidc_ndl.to_csv(f'{path_dest}lidc info/{f_name[:-3]}csv', index=False)
        cube_orig.tofile(f'{path_dest}original/{f_name}')
        cube_inpain.tofile(f'{path_dest}inpainted inserted/{f_name}')
        mask_ndl.tofile(f'{path_dest}mask/{f_name}')

