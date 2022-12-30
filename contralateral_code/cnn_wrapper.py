import os
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate

import file_structure
from scan_handler import norm_modals, default_modal_list, default_modal_dict
from keras.models import load_model, Model
# import gc

def load_model_set(modal_list = default_modal_list()):
    # last layer
    layer_name = 'flatten_1'
    #Specify which GPU to be used
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    (origdir,basedir,imagedir,normdir,splitdir,out_dir,model_dir) = file_structure.file_dirs('CNN')
#     pre_trained_dir = os.path.join(origdir,'..','glioma_models')
    pre_trained_dir = os.path.join(basedir,'glioma_models')
    model_set = {}
    for modal in modal_list:
        pre_trained_model_name = os.path.join(pre_trained_dir, modal + '_model.h5')
#         print(pre_trained_model_name)

        pre_trained_model = load_model(pre_trained_model_name)
        model_FC_layer_output = Model(inputs=pre_trained_model.input, outputs=pre_trained_model.get_layer(layer_name).output)
        model_set[modal] = model_FC_layer_output
    return model_set

def get_cnn_feats(cDir,label_arr = None, label_fname = 'truth.nii.gz', feat_suffix = '', min_size = 10,modal_list = default_modal_list(), modal_dict = default_modal_dict(),model_set = None):
    # input is directory, output is array of features
    # operational notes about this ML model, it doesn't care about the segmentation really
    # it takes in axial, sagittal and coronal cuts of the tumor
    
    img = norm_modals(cDir,modal_list = modal_list, modal_dict= modal_dict)
    # sagittal slices, axial slices,  coronal slices
    dim_arr = ((2,0,1),(0,1,2),(1,2,0))
    desired_size = (142,142)
    
    
    img = norm_modals(cDir,modal_list = modal_list, modal_dict = modal_dict)
    if model_set is None:
        model_set = load_model_set(modal_list = modal_list)
    
    cLblFile = os.path.join(cDir,label_fname)
    if not os.path.exists(cLblFile):
        return np.array([]), []
    label_img = nib.load(cLblFile)
    
    if label_arr is None:
        label_arr = label_img.get_data().astype(np.float64)

    data_spacing = label_img.header['pixdim'][1:4]
    mask = (label_arr >= 1) * 1
    
    features = {}
    for modal in modal_list:
        features[modal] = {}
        if len(img[modal]) == 0:
            continue
        # [n_slices, x_dim, y_dim, views]
        mask_ct = np.sum(mask,axis=(0,1))
        z_slice_vec = np.argwhere(mask_ct >= min_size).flatten()
        z_ct = len(z_slice_vec)
        n_slices = z_ct
        
        slice_set = np.zeros((n_slices,desired_size[0],desired_size[1],3))
        mask_set = np.zeros((n_slices,desired_size[0],desired_size[1],3))
        for dN in (1,0,2):
            # crop the image based on the mask bounding box
            B = np.argwhere(mask)
            (start_i, start_j, _), (stop_i, stop_j, _) = B.min(0), B.max(0) + 1  
            
            # transpose the images to set up for the three part network
            tImg = np.transpose(img[modal][start_i:stop_i, start_j:stop_j,z_slice_vec],axes = dim_arr[dN])
            tMask = np.transpose(mask[start_i:stop_i, start_j:stop_j,z_slice_vec],axes = dim_arr[dN])
            
#             print(tMask.shape)
            # count the # of cancer pixels per slice
            m_sum = np.sum(tMask>0,axis=(0,1))

            # find the slices in a direction with pixels
            slice_vec = np.argwhere(m_sum >= min_size).flatten()
            count_idxs = len(slice_vec)
#             print(slice_vec)
            
            dim = img[modal].shape
            if len(dim)>=3:
                n_slices = len(slice_vec)
            else:
                n_slices = 1
                
            if dN == 1: # has to be the vector indices after cropping, I could do it before the loop, but this is easier.
                zInds = slice_vec

            for slI in range(n_slices):
                slice_ind = slice_vec[slI]
                # find where among the slices of the mask the current slice is for it's dimension
                match_inds = np.argwhere(np.equal(zInds,slice_ind))
#                 print(np.equal(zInds,slice_ind))
                if len(match_inds) == 0:
#                     print("%i not %i" % (dN,slice_ind))
                    continue
                match_idx = match_inds[0]
                # goal is to find the xth percent slice in a direction, if the slice_num is xth percent in location
                nearest_idx = int((match_idx * count_idxs) / z_ct)

                c_idx = slice_vec[nearest_idx]
#                 print(slice_ind,c_idx)

                # take slices from the transposed image
                tImg_slice = tImg[:,:,c_idx]
                tMask_slice = tMask[:,:,c_idx]
            
                # pick slice and view
                mask_orig = tMask[:,:,slice_ind] == 1
                if np.sum(mask_orig,axis=(0,1)) < min_size:
                    features[modal][slI]=[] # should be caught earlier
                    continue
                # crop the image
                B = np.argwhere(tMask_slice)
                irange = np.empty((2), dtype=int)
                jrange = np.empty((2), dtype=int)
                (irange[0], jrange[0]), (irange[1], jrange[1]) = B.min(0), B.max(0) + 1  

                if (irange[1]-irange[0] <= 1) or (jrange[1]-jrange[0] <= 1):
                    features[modal][slI]=[]
                    continue 
#                 print(irange,jrange)
                # crop the image by tumor on the slice
                img_slice  =  tImg_slice[irange[0]:irange[1], jrange[0]:jrange[1]]
                mask_slice = tMask_slice[irange[0]:irange[1], jrange[0]:jrange[1]]


                # this is the original schema for sampling
                # cropped exactly to the tumor
                xcrIV = np.linspace(jrange[0],jrange[1],desired_size[1])
                ycrIV = np.linspace(irange[0],irange[1],desired_size[0])
                (xcrm,ycrm) = np.meshgrid(xcrIV, ycrIV)
                cr_im = ndimage.map_coordinates(img_slice,[ycrm, xcrm])
                # sample the mask at the same locations
                # order = 0 is to make it categorical in its output (not 100% that's the setting, I know how to do it in MATLAB)
                cr_mask = ndimage.map_coordinates(mask_slice,[ycrm, xcrm],order = 0)
                slice_set[slice_ind,:,:,dN] = cr_im
                mask_set[slice_ind,:,:,dN] = cr_mask

        model_FC_layer_output = model_set[modal]
        features[modal] = model_FC_layer_output.predict(slice_set)
        
    feat_names = []
    for midx in range(len(modal_list)):
        modal = modal_list[midx]
        # load data
        nFeats = model_set[modal].output_shape[1]
        cFeatNames = ['CNN_%03d_%s%s' % (x,modal,feat_suffix) for x in range(nFeats)]
        if img[modal].size == 0: # handle empty modalities (fill with nan)
            mFeats = np.empty((n_slices,nFeats))
            mFeats[:] = np.nan
        else: # otherwise just use the features collected
            mFeats = features[modal]
        
        if midx == 0:
            feature_arr = mFeats
            feat_names = np.asarray(cFeatNames)
        else:
            feature_arr = np.append(feature_arr, mFeats, axis=1)
            feat_names = np.append(feat_names,cFeatNames,axis=0)
            
    # gc.collect() # add garbage collection to handle TF kernel crashes
    return feature_arr,feat_names