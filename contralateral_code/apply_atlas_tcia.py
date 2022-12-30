import os
import gc

import numpy as np
import pandas as pd
import pickle
from scipy.stats import iqr
from scipy.ndimage import morphology
from copy import deepcopy

from datetime import datetime

#import globz
import file_structure, tex_wrapper, tda_wrapper, cnn_wrapper
from nilearn import datasets, image, plotting
import nibabel as nib
import matplotlib.pyplot as plt

from dipy.align import vector_fields as vf
from dipy.viz import regtools
from dipy.align.imaffine import (AffineMap,MutualInformationMetric,AffineRegistration,transform_centers_of_mass)
from dipy.align.transforms import (TranslationTransform3D,RigidTransform3D,AffineTransform3D)
from dipy.align.reslice import reslice

from mri_reg import ms_affreg, resize_template,vox_coords
(origdir,basedir,imagedir,normdir,splitdir,out_dir,model_dir) = file_structure.file_dirs('texture')

modal_list = ['FLAIR','T1', 'T1post','T2']
modal_dict = {'t1': 't1.nii.gz','t1post':'t1Gd.nii.gz','t2':'t2.nii.gz','flair': 'flair.nii.gz'}

os.chdir(imagedir)

patients = []
for root, dirs, files in os.walk('.'):
    for dn in dirs:
        patients.append(os.path.join(root,dn))
os.chdir(origdir)

outImDir = os.path.join(basedir,'data','images','atlases')
if not os.path.exists(outImDir):
    os.makedirs(outImDir)

### https://nilearn.github.io/modules/generated/nilearn.datasets.load_mni152_template.html
# template_img = datasets.load_mni152_template()
orig_template = datasets.load_mni152_template()
template_img = orig_template
# orig_zooms = orig_template.header.get_zooms()[:3]
rerunAffine = False
model_set = cnn_wrapper.load_model_set()

# for pNo in range(131,len(patients)):
for pNo in range(0,len(patients)):
    pDir = patients[pNo]
    cDir = os.path.join(imagedir,pDir)
    cNIIFile = os.path.join(cDir,'t1.nii.gz')
    if not os.path.exists(cNIIFile):
        continue
    TCGA_ID = os.path.basename(pDir)

    current_time = datetime.now().strftime("%m/%d/%y %H:%M:%S")
    print('%s- %s' % (current_time,TCGA_ID))
    

    moving_img = nib.load(cNIIFile)
    if template_img.shape != moving_img.shape:
        print('Resizing template from: ', template_img.shape, ' to ', moving_img.shape)
        template_img = resize_template(orig_template,moving_img)
    print(pDir)    
    cLblFile = os.path.join(cDir,'truth.nii.gz')
    if not os.path.exists(cLblFile):
        continue
    label_img = nib.load(cLblFile)
    label_arr = label_img.get_fdata().astype(np.float64)
    moving_arr2 = deepcopy(moving_img.get_fdata())
    bMask = np.logical_and(moving_arr2 > 0,label_arr == 0)
    medianVal = np.median(moving_arr2[bMask])
    moving_arr2[np.where(label_arr>0)] = medianVal
    # run the multi-step registration (mri_reg.py)
    # (affine,_) = ms_affreg(moving_img,template_img=template_img)
#     moving_zooms = moving_img.header.get_zooms()[:3]
#     temp_rz, temp_aff_rz = reslice(orig_template.get_data(), orig_template.affine, orig_zooms, moving_zooms)
# #     header = orig_template.header
#     template_img = nib.nifti1.Nifti1Image(temp_rz,temp_aff_rz,header = orig_template.header)
#     template_img = orig_template
#     print(template_img)
#     print(template_img.shape,moving_img.shape)
#     print('MNI')
#     f, ax = plt.subplots(figsize=(15, 10))
#     plotting.plot_anat(template_img,axes=ax,cut_coords =(0,0,0))
#     plt.show()
    
    # run the multi-step registration (mri_reg.py)
    cTrMapFile = os.path.join(cDir,'affine_map_2.pkl')
    if rerunAffine or not os.path.isfile(cTrMapFile):
        (affine,_) = ms_affreg(moving_img2,template_img=template_img,reg_type=3)
        f = open(cTrMapFile,"wb")
        pickle.dump(affine,f)
        f.close()
    else:
        f = open(cTrMapFile, 'rb')
        affine = pickle.load(f)
        f.close()

    
    transformed = affine.transform(moving_img.get_fdata(),interp = 'nearest')
    transf_nib = nib.nifti1.Nifti1Image(transformed,template_img.affine,header=moving_img.header)
#     tr_same_sz = affine.transform(moving_img.get_data(),sampling_grid_shape=moving_img.shape)
#     transf_nib = nib.nifti1.Nifti1Image(tr_same_sz,template_img.affine,header=moving_img.header)
    # show the sag,cor,axi alignment map
#     regtools.overlay_slices(template_img.get_data(), transformed, None, 0,
#                         "Template", "Transformed")
#     regtools.overlay_slices(template_img.get_data(), transformed, None, 1,
#                             "Template", "Transformed")
#     regtools.overlay_slices(template_img.get_data(), transformed, None, 2,
#                             "Template", "Transformed")
    
#     print('MNI')
#     f, ax = plt.subplots(figsize=(15, 10))
#     plotting.plot_anat(template_img,axes=ax,cut_coords =(0,0,0))
#     plt.show()
#     f, ax = plt.subplots(figsize=(9, 6))
#     plotting.plot_anat(transf_nib,axes=ax,cut_coords =(0,0,0))
#     plt.savefig(os.path.join(outImDir,'affreg_%s.png' % (TCGA_ID)))
#     plt.show()
    
#     bmask_nib = nib.nifti1.Nifti1Image((transformed > 0)*1,template_img.affine,header=moving_img.header)

#     f, ax = plt.subplots(figsize=(9, 6))
#     plotting.plot_anat(bmask_nib,axes=ax,cut_coords =(0,0,0))
#     plt.savefig(os.path.join(outImDir,'affreg_%s.png' % (TCGA_ID)))
#     plt.show()
    
    
    # contralateral side of tumor

    tr_label = np.round(affine.transform(label_arr,interp = 'nearest'),decimals=0)
#     regtools.overlay_slices(moving_img.get_data(), label_img.get_data(), None, 2,
#                             "Orig T1", "Orig Label")
#     regtools.overlay_slices(template_img.get_data(), tr_label, None, 2,
#                             "Transformed T1", "Transformed Label")

    refl_x_aff_arr = nib.affines.from_matvec(np.diag([-1, 1, 1]), [0,0,0])
    refl_x_affine = AffineMap(refl_x_aff_arr,
                           tr_label.shape, template_img.affine,
                           tr_label.shape, template_img.affine)

    refl_label = np.round(refl_x_affine.transform(tr_label,interp = 'nearest'),decimals=0)
    # print(refl_label)
#     regtools.overlay_slices(template_img.get_data(), refl_label, None, 2,
#                             "Transformed T1", "Contralateral Label")
#     transf_label_nib = nib.nifti1.Nifti1Image(tr_label,template_img.affine,header=moving_img.header)
#     refl_label_nib = nib.nifti1.Nifti1Image(refl_label,template_img.affine,header=moving_img.header)

#     f, ax = plt.subplots(figsize=(9, 6))
#     pl_anat = plotting.plot_anat(transf_nib,axes=ax,cut_coords =(0,0,0))
#     pl_anat.add_overlay(transf_label_nib)
#     plt.show()
#     f, ax = plt.subplots(figsize=(15, 10))
#     pl_anat = plotting.plot_anat(transf_nib,axes=ax,cut_coords =(0,0,0))
#     pl_anat.add_overlay(refl_label_nib)
#     plt.show()


#     cent_coords = nib.affines.apply_affine(-np.linalg.inv(affine.affine),(0,0,0))
    contr_label = affine.transform_inverse(refl_label,image_grid2world=template_img.affine,
                                           sampling_grid2world=moving_img.affine,sampling_grid_shape = moving_img.shape,
                                          interp = 'nearest')
    contr_label = contr_label.astype(int)
    contr_label_nib = nib.nifti1.Nifti1Image(contr_label,moving_img.affine,header=moving_img.header)
    contr_xyz = plotting.find_xyz_cut_coords(moving_img, mask_img=contr_label_nib, activation_threshold=None)
    label_xyz = plotting.find_xyz_cut_coords(moving_img, mask_img=label_img, activation_threshold=None)
#     print(contr_xyz,label_xyz)
    cent_coords = np.add(contr_xyz,label_xyz) /2
    
#     f, ax = plt.subplots(figsize=(9, 6))
#     pl_anat = plotting.plot_anat(moving_img,axes=ax,cut_coords=cent_coords)
#     pl_anat.add_overlay(contr_label_nib)
#     pl_anat.add_overlay(label_img)
#     plt.show()
    
    
    tum_arr = np.logical_and(label_arr,moving_img.get_fdata() > 0) # exclude regions outside of brain mask
    cl_arr = np.logical_and(contr_label,moving_img.get_fdata() > 0) # exclude regions outside of brain mask
    
    tum_cl_iou = np.count_nonzero(np.logical_and(tum_arr,cl_arr))/np.count_nonzero(np.logical_or(tum_arr,cl_arr))

    stopRun = False
    if np.sum(np.logical_and(contr_label>=1,label_arr>=1)) > 0:
        print('overlapping labels')
        print(np.count_nonzero(np.logical_and(tum_arr,cl_arr)),np.count_nonzero(np.logical_or(tum_arr,cl_arr)))
        print(tum_cl_iou)
        if tum_cl_iou > 0.05: # 5% threshold
            print('very overlapping labels')
            stopRun = True
            for ftType in ('tex','cnn','tda'):
                save_str = ('%s_%s.csv' % (ftType,'feats'))
                save_fname = os.path.join(cDir,save_str)
                if os.path.isfile(save_fname):
                    print('deleting %s' % save_fname)
                    os.remove(save_fname)
                
    if stopRun:
        continue
    clnt_arr =  np.logical_and(cl_arr,np.logical_not(tum_arr)) # exclude tumor region from contralateral side
    
    # Texture Features
    current_time = datetime.now().strftime("%m/%d/%y %H:%M:%S")
    print('%s- Starting texture' % current_time)
    (tum_samples,tum_feature_names) = tex_wrapper.get_tex_feats(cDir, label_arr = tum_arr, feat_suffix='_tum')
    (cl_samples,cl_feature_names) = tex_wrapper.get_tex_feats(cDir, label_arr = clnt_arr, feat_suffix='_cl')

    tum_df = pd.DataFrame(data=tum_samples, columns = tum_feature_names)
    cl_df = pd.DataFrame(data=cl_samples, columns = cl_feature_names)
    tum_df['sliceNo'] = tum_df.index
    cl_df['sliceNo'] = cl_df.index
    data_frame = pd.merge(tum_df, cl_df, on='sliceNo', how='outer')
    save_str = ('tex_%s.csv' % 'feats')
    save_fname = os.path.join(cDir,save_str)
    data_frame.to_csv(save_fname)
    
    # CNN Features
    current_time = datetime.now().strftime("%m/%d/%y %H:%M:%S")
    print('%s- Starting CNN' % current_time)
    (tum_samples,tum_feature_names) = cnn_wrapper.get_cnn_feats(cDir, label_arr = tum_arr, feat_suffix='_tum', model_set = model_set)
    (cl_samples,cl_feature_names) = cnn_wrapper.get_cnn_feats(cDir, label_arr = clnt_arr, feat_suffix='_cl', model_set = model_set)
    
    tum_df = pd.DataFrame(data=tum_samples, columns = tum_feature_names)
    cl_df = pd.DataFrame(data=cl_samples, columns = cl_feature_names)
    tum_df['sliceNo'] = tum_df.index
    cl_df['sliceNo'] = cl_df.index
    data_frame = pd.merge(tum_df, cl_df, on='sliceNo', how='outer')
    save_str = ('cnn_%s.csv' % 'feats')
    save_fname = os.path.join(cDir,save_str)
    data_frame.to_csv(save_fname) 
    
    # TDA Features
    current_time = datetime.now().strftime("%m/%d/%y %H:%M:%S")
    print('%s- Starting TDA' % current_time)
    (tum_samples,tum_feature_names) = tda_wrapper.get_tda_feats(cDir, label_arr = tum_arr, feat_suffix='_tum')
    (cl_samples,cl_feature_names) = tda_wrapper.get_tda_feats(cDir, label_arr = clnt_arr, feat_suffix='_cl')
    
    tum_df = pd.DataFrame(data=tum_samples, columns = tum_feature_names)
    cl_df = pd.DataFrame(data=cl_samples, columns = cl_feature_names)
    tum_df['sliceNo'] = tum_df.index
    cl_df['sliceNo'] = cl_df.index
    data_frame = pd.merge(tum_df, cl_df, on='sliceNo', how='outer')
    save_str = ('tda_%s.csv' % 'feats')
    save_fname = os.path.join(cDir,save_str)
    
    current_time = datetime.now().strftime("%m/%d/%y %H:%M:%S")
    print('%s- Saving: %s' % (current_time,save_fname))
    
    data_frame.to_csv(save_fname) 
    gc.collect()
#     samples = np.concatenate((tum_samples, cl_samples), axis=1)
#     feat_names = tum_feature_names + cl_feature_names
#     data_frame.to_csv(save_fname)
    