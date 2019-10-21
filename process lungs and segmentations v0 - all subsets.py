import sys
print(sys.path)
print(sys.executable)
sys.path.append('/home/user/anaconda3/envs/luna/lib/python3.6/site-packages')

##

import SimpleITK as sitk
import numpy as np
import csv
import os
from PIL import Image
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.sparse
import scipy.spatial
import pandas as pd
from scipy import ndimage
import matplotlib.patches as patches
import pylidc as pl
from pylidc.utils import consensus
import time
from skimage.morphology import ball, dilation

##

from scipy import sparse
get_ipython().run_line_magic('matplotlib', 'inline')

##

data_dir  = '/data/OMM/Datasets/LUNA/'
cand_path = '/data/OMM/Datasets/LUNA/CSVFILES/candidates_V2.csv'
out_path = '/data/OMM/Datasets/LUNA/candidates/'
annotations_path = '/data/OMM/Datasets/LUNA/CSVFILES/annotations.csv'
data_seg = f'{data_dir}seg-lungs-LUNA16/'
# if not os.path.exists(out_path): os.makedirs(out_path)

##

def resample_sitk(image,spacing, new_spacing=[1,1,1]):    

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
    
    # make the reference image grid, 80x80x80, with new spacing
    refImg = sitk.GetImageFromArray(np.zeros((80,80,80),dtype=image.dtype))
    refImg.SetSpacing(new_spacing_sitk)
    refImg.SetOrigin(img.GetOrigin())
    
    imgNew = sitk.Resample(img, refImg, affine,sitk.sitkLinear,0)
    
    imOut = sitk.GetArrayFromImage(imgNew).copy()
    
    return imOut

##

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

def resample_scan_sitk_neighbors(image,spacing, original_shape, new_spacing=[1,1,1]):    

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
    
    imgNew = sitk.Resample(img, refImg, affine, sitk.sitkNearestNeighbor, 0)
    
    imOut = sitk.GetArrayFromImage(imgNew).copy()
    
    return imOut

##

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing

def readCSV(filename):
    lines = []
    with open(filename, "rt") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

def worldToVoxelCoord(worldCoord, origin, spacing):
     
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord // spacing
    return [int(i) for i in voxelCoord]

def normalizePatches(npzarray):
    npzarray = npzarray
    
    maxHU = 400.
    minHU = -1000.
 
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

##

def median_center(segmentation):
    z,y,x = np.where(segmentation==1)
    zz = np.median(z)
    yy = np.median(y)
    xx = np.median(x)
    return zz,yy,xx

##

def make3d_from_sparse(path):
    slices_all = os.listdir(path)
    slices_all = np.sort(slices_all)
    for idx, i in enumerate(slices_all):
        sparse_matrix = sparse.load_npz(f'{path}{i}')
        array2d = np.asarray(sparse_matrix.todense())
        if idx == 0: 
            scan3d = array2d
            continue
        scan3d = np.dstack([scan3d,array2d])
    return scan3d

##

def make3d_from_sparse_v2(path):
    slices_all = os.listdir(path)
    slices_all = np.sort(slices_all)
    for idx, i in enumerate(slices_all):
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

# Added in v2, df from pylidc
df = pd.read_csv('/data/datasets/LIDC-IDRI/annotations.csv')

##

# load candidates
cands = readCSV(cand_path)
cands_df = pd.read_csv(cand_path)
# load annotations
annotations_df = pd.read_csv(annotations_path)
subset_dir = [f'subset{str(ii)}' for ii in range(10)]

##

# EACH SUBSET
for kk in range(0,10):
    curr_dir = subset_dir[kk]
    if not os.path.exists(out_path + curr_dir): os.makedirs(out_path + curr_dir)

    subset_files = os.listdir(data_dir + curr_dir)
    subset_series_ids = np.unique(np.asarray([subset_files[ll][:-4:] for ll in range(len(subset_files))]))
    subset_series_ids = np.sort(subset_series_ids)
    
    # initialise an output dataframe with the nodule size (nan for non-nodules)
    out_df = pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','class','diameter_mm','filename'])

    columns_plus_new_coords = np.append(df.columns.values,['coordZ_cands_class1_resampled', 'coordY_cands_class1_resampled', 'coordX_cands_class1_resampled'])
    columns_plus_new_coords_class1 = np.append(cands_df.columns.values,['coordX_resamp', 'coordY_resamp', 'coordZ_resamp'])

    # EACH IMAGE
    for jj in range(len(subset_series_ids)):
        
        print(f'subset = {kk}, id = ({jj}) {subset_series_ids[jj]}')
#         iii=11
#         if jj < iii:continue
#         if jj == iii+3:break
        # initialize output dataframes
        df_lidc_new = pd.DataFrame(columns=columns_plus_new_coords)
        df_cands_class1 = pd.DataFrame(columns=columns_plus_new_coords_class1)

        image_file = data_dir + curr_dir + '/' + subset_series_ids[jj] + '.mhd'
        numpyImage, numpyOrigin, numpySpacing = load_itk_image(image_file)

        curr_cands = cands_df.loc[cands_df['seriesuid'] == subset_series_ids[jj]].reset_index(drop=True)

        # resample original image
        new_spacing = [1,1,1]
        numpyImage_shape = ((np.shape(numpyImage) * numpySpacing) / np.asarray(new_spacing)).astype(int)
        numpyImage_resampled = resample_scan_sitk(numpyImage, numpySpacing, numpyImage_shape, new_spacing=new_spacing)
        # get the correspponding lung segmentation
        path_segment_lungs = f'{data_seg}{subset_series_ids[jj]}.mhd'
        segment_lungs, seglung_orig, seglung_spacing = load_itk_image(path_segment_lungs)
        assert (numpyOrigin - seglung_orig < .1).all()
        assert (numpySpacing - seglung_spacing < .1).all()
        segment_lungs_resampled = resample_scan_sitk(segment_lungs, numpySpacing, numpyImage_shape, new_spacing=new_spacing)

        # normalize
        numpyImage_normalized = normalizePatches(numpyImage_resampled)

        # save the segmented lungs
        numpyImage_segmented = numpyImage_normalized * (segment_lungs_resampled>0)
        if not os.path.exists(f'{out_path}{subset_series_ids[jj]}/lungs_segmented'): os.makedirs(f'{out_path}{subset_series_ids[jj]}/lungs_segmented')
        np.savez_compressed(f'{out_path}{subset_series_ids[jj]}/lungs_segmented/lungs_segmented.npz',numpyImage_segmented)

        # go through all candidates that are in this image
        # sort to make sure we have all the trues (for prototyping only)
        curr_cands = curr_cands.sort_values('class',ascending=False).reset_index(drop=True)

        # Added in v2
        one_segmentation_consensus = np.zeros_like(numpyImage)
        one_segmentation_maxvol = np.zeros_like(numpyImage)
        labelledNods = np.zeros_like(numpyImage)

        # query the LIDC images HERE WE JUST USE THE FIRST ONE!!
        idx_scan = 0 
        scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == subset_series_ids[jj])[idx_scan] 
        nods = scan.cluster_annotations() # get the annotations for all nodules in this scan
        #print(np.shape(nods))

        #Get all the nodules (class==1)
        curr_cands_class1 = curr_cands.loc[curr_cands['class']==1]

        for i_curr_cand in range(len(curr_cands_class1)):
            curr_cand = curr_cands_class1.iloc[i_curr_cand]
            # first need to find the corresponding column in the annotations csv (assuming its the closest 
            # nodule to  the current candidate)
            # extract the annotations for the scan id of our current candidate
            annotations_scan_df = annotations_df.loc[annotations_df['seriesuid'] == curr_cand['seriesuid']]

            # Get the right coordinates        
            worldCoord = [curr_cand['coordZ'], curr_cand['coordY'], curr_cand['coordX']]
            voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
            voxelCoord_resampled = ((voxelCoord * numpySpacing) / np.asarray(new_spacing)).astype(int)

            # Create new DF with all curr_cands['class']==1
            curr_cand['coordZ_resamp'] = voxelCoord_resampled[0]
            curr_cand['coordY_resamp'] = voxelCoord_resampled[1]
            curr_cand['coordX_resamp'] = voxelCoord_resampled[2]
            df_cands_class1 = df_cands_class1.append(curr_cand)

            df_nodule = df.loc[df['seriesuid'] == curr_cand['seriesuid']]

            # SAVE DATAFRAMES (the pylidc DF from the nodules in LUNA)
            threshold_coord_found  = 4
            # seriesuid might include several nodules, if its coords are close to the coords in annotations then save
            df_series_all_nodules = df.loc[df['seriesuid']==subset_series_ids[jj]]
            df_series_number_nodules = np.unique(df_series_all_nodules['cluster_id'].values)
            for idxx, i_number_nodule in enumerate(df_series_number_nodules):
                df_series_one_nodule = df_series_all_nodules.loc[df_series_all_nodules['cluster_id']==i_number_nodule]
                lidc_coordZ = np.mean(df_series_one_nodule['lidc_coordZ'].values)
                lidc_coordY = np.mean(df_series_one_nodule['lidc_coordY'].values)
                lidc_coordX = np.mean(df_series_one_nodule['lidc_coordX'].values)
    #             print(f'lidc_coords = {lidc_coordZ, lidc_coordX, lidc_coordY}')
    #             print(f'voxel_coords = {voxelCoord, voxelCoord_resampled}')
    #             print(np.sum(np.abs(np.asarray([lidc_coordZ, lidc_coordX, lidc_coordY]) - voxelCoord)))
                # WARNING: now the comparison is done Z-Z, X-Y, Y-X instead of Z-Z, Y-Y, X-X
                if np.sum(np.abs(np.asarray([lidc_coordZ, lidc_coordX, lidc_coordY]) - voxelCoord)) < threshold_coord_found:
    #                 print(f'save')
                    df_series_one_nodule_save = df_series_one_nodule
                    df_series_one_nodule_save['coordZ_cands_class1_resampled'] = voxelCoord_resampled[0]
                    df_series_one_nodule_save['coordY_cands_class1_resampled'] = voxelCoord_resampled[1]
                    df_series_one_nodule_save['coordX_cands_class1_resampled'] = voxelCoord_resampled[2]
                    df_lidc_new = df_lidc_new.append(df_series_one_nodule_save)

        # save dataframes
        if os.path.isfile(f'{data_dir}dataframes/df_cands_class1.csv'):
            df_cands_class1.to_csv(f'{data_dir}dataframes/df_cands_class1.csv', index=False, mode='a', header=False)
            df_lidc_new.to_csv(f'{data_dir}dataframes/df_lidc_new.csv', index=False, mode='a', header=False)

        else:
            df_cands_class1.to_csv(f'{data_dir}dataframes/df_cands_class1.csv', index=False, mode='w', header=True)
            df_lidc_new.to_csv(f'{data_dir}dataframes/df_lidc_new.csv', index=False, mode='w', header=True)

        # SAVE SEGMENTATIONS    TO DO!!!! ADD FOLDER NAMES FOR CASES WHEN VARIOUS SEGMENTATIONS ARE SAVED
        for idx_anns, anns in enumerate(nods): 
    #             if idx_anns==1:break # WARNING !!!!!!!
            #print(idx_anns)
            cmask,cbbox,masks = consensus(anns, clevel=0.5, pad=[(0,0), (0,0), (0,0)])

            # we want to save the consensus AND the mask of all segmented voxels in all annotations
            one_mask_consensus = cmask
            one_mask_maxvol = np.zeros_like(cmask)
            for mask in masks:
                one_mask_maxvol = (one_mask_maxvol > 0) | (mask > 0)    

            # pylidc loads in a different order to our custom 3D dicom reader, so need to swap dims
            one_mask_consensus = np.swapaxes(one_mask_consensus,1,2);one_mask_consensus = np.swapaxes(one_mask_consensus,0,1)
            one_mask_maxvol = np.swapaxes(one_mask_maxvol,1,2);one_mask_maxvol = np.swapaxes(one_mask_maxvol,0,1)

            # Dilate the mask
            one_mask_maxvol = dilation(one_mask_maxvol)

            # fill the consensus bounding box with the mask to get a nodule segmentation in original image space (presumably the cbbox is big enough for all the individual masks)
            one_segmentation_consensus[cbbox[2].start:cbbox[2].stop,cbbox[0].start:cbbox[0].stop,cbbox[1].start:cbbox[1].stop] = one_mask_consensus
            one_segmentation_maxvol[cbbox[2].start:cbbox[2].stop,cbbox[0].start:cbbox[0].stop,cbbox[1].start:cbbox[1].stop] = one_mask_maxvol
            labelledNods[cbbox[2].start:cbbox[2].stop,cbbox[0].start:cbbox[0].stop,cbbox[1].start:cbbox[1].stop] = one_mask_maxvol * (idx_anns + 1) # label each nodule with its 'cluster_id'

            #print(np.sum(one_segmentation_consensus))
            # find the center of this segmentation. If it is close to the TRUE NODULE then save
            #zz_seg,yy_seg,xx_seg = median_center(one_segmentation_consensus)
            #print(f'segmentation_consensus = {zz_seg,yy_seg,xx_seg}')
            #print(f'voxelCoord = {voxelCoord, voxelCoord_resampled}')
            #print(np.abs(np.sum(np.asarray([zz_seg,yy_seg,xx_seg]) - voxelCoord)))
            # we save all the segmentations. Anyway they dont affect inpainting because they are very small
            #if np.abs(np.sum(np.asarray([zz_seg,yy_seg,xx_seg]) - voxelCoord)) < threshold_coord_found:
            #    print('save segment')
            # RESAMPLE
        one_segmentation_consensus_resampled = resample_scan_sitk(one_segmentation_consensus, numpySpacing, numpyImage_shape, new_spacing, sitk.sitkNearestNeighbor)
        one_segmentation_maxvol_resampled = resample_scan_sitk(one_segmentation_maxvol, numpySpacing, numpyImage_shape, new_spacing, sitk.sitkNearestNeighbor)
        labelledNods_resampled = resample_scan_sitk(labelledNods, numpySpacing, numpyImage_shape, new_spacing, sitk.sitkNearestNeighbor)

        if not os.path.exists(f'{out_path}{subset_series_ids[jj]}/consensus_masks'): os.makedirs(f'{out_path}{subset_series_ids[jj]}/consensus_masks')
        if not os.path.exists(f'{out_path}{subset_series_ids[jj]}/maxvol_masks'): os.makedirs(f'{out_path}{subset_series_ids[jj]}/maxvol_masks')
        if not os.path.exists(f'{out_path}{subset_series_ids[jj]}/cluster_id_images'): os.makedirs(f'{out_path}{subset_series_ids[jj]}/cluster_id_images')

        for i_sparse, (one_seg_consen, one_seg_max, labelNods) in enumerate(zip(one_segmentation_consensus_resampled, one_segmentation_maxvol_resampled, labelledNods_resampled)):
            sparse_matrix_one_segmentation_consensus = scipy.sparse.csc_matrix(one_seg_consen)
            sparse_matrix_one_segmentation_maxvol = scipy.sparse.csc_matrix(one_seg_max)
            sparse_matrix_labelledNods = scipy.sparse.csc_matrix(labelNods)

            scipy.sparse.save_npz(f'{out_path}{subset_series_ids[jj]}/consensus_masks/slice_{i_sparse:04d}.npz', sparse_matrix_one_segmentation_consensus, compressed=True)
            scipy.sparse.save_npz(f'{out_path}{subset_series_ids[jj]}/maxvol_masks/slice_m_{i_sparse:04d}.npz', sparse_matrix_one_segmentation_maxvol, compressed=True)
            scipy.sparse.save_npz(f'{out_path}{subset_series_ids[jj]}/cluster_id_images/slice_m_{i_sparse:04d}.npz', sparse_matrix_labelledNods, compressed=True)

