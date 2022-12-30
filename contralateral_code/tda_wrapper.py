import os, math
import numpy as np
import nibabel as nib
from gudhi import CubicalComplex as CC
from gudhi import plot_persistence_barcode as plt_bar
from scan_handler import norm_modals, default_modal_list, default_modal_dict
from copy import deepcopy

def default_feat_parts():
    feat_parts = ['polynomial_1',
             'polynomial_2', 
             'polynomial_3', 
             'polynomial_4',
             'mean(birth)', 
             'mean(death)', 
             'mean(death_max-death)',
             'mean(bar length)',
             'median(birth)', 
             'median(death)', 
             'median(death_max-death)',
             'median(bar length)',
             'std(birth)', 
             'std(death)', 
             'std(bar length)']
    return feat_parts
def default_level_list():
    return ['0', '1']
def default_label_fname():
    return 'truth.nii.gz'
def default_min_size():
    return 10 # slices smaller than this are excluded
def default_non_roi_val():
    return 1000. # values outside of the ROI are set to this
def default_lim():
    return 1.1 # if the diagonal has length(0), replace the feature value with this #
# def default_modal_list():
#     return ['FLAIR','T1', 'T1post','T2']
# def default_modal_dict():
#     return {'t1': 't1.nii.gz', 't1post':'t1Gd.nii.gz', 't2':'t2.nii.gz', 'flair': 'flair.nii.gz'}


def f_1(diag):
	return sum( diag[:,1] * (diag[:,2] - diag[:,1]) ) / len(diag)

def f_2(diag):
	return sum( (diag[:,2].max() - diag[:,2]) * (diag[:,2] - diag[:,1]) )/ len(diag)

def f_3(diag):
	return sum( pow(diag[:,1],2) * pow( (diag[:,2] - diag[:,1]) ,4) )/ len(diag)

def f_4(diag):
	return sum( pow( (diag[:,2].max() - diag[:,2] ), 2) * pow( (diag[:,2] - diag[:,1]), 4) )/ len(diag)

def g_1(diag):
	return diag[:,1].mean()

def g_2(diag):
	return diag[:,2].mean()

def g_3(diag):
	return ( diag[:,2].max() - diag[:,2] ).mean()

def g_4(diag):
	return ( diag[:,2] - diag[:,1] ).mean()

def g_5(diag):
	return np.median(diag[:,1])

def g_6(diag):
	return np.median(diag[:,2])

def g_7(diag):
	return np.median(diag[:,2].max() - diag[:,2])

def g_8(diag):
	return np.median(diag[:,2] - diag[:,1])

def g_9(diag):
	return diag[:,1].std()

def g_10(diag):
	return diag[:,2].std()

def g_11(diag):
	return (diag[:,2] - diag[:,1]).std()



def get_feature_names(modal_list,suffix = '', feature_parts = default_feat_parts(),level_list =default_level_list()):
    ## Feature names
    feature_names_level = []
    feature_names_all =[]

    for level in level_list:
        feature_names_level.append(['%s_%s' % (elemt, level) for elemt in feature_parts])

    feature_names_level = [val for sublist in feature_names_level for val in sublist]

    for modal in modal_list:
        feature_names_all.append(['%s_%s%s' % (elemt, modal, suffix)  for elemt in feature_names_level])

    feature_names = [val for sublist in feature_names_all for val in sublist]
    return feature_names, feature_names_level

def get_diag_dict(cDir, modal_list = default_modal_list(), modal_dict = default_modal_dict(),label_fname = default_label_fname(),label_arr = None,min_size = default_min_size(),non_roi_val = default_non_roi_val(), lim = default_lim()):
    
    img = norm_modals(cDir,modal_list = modal_list, modal_dict= modal_dict)
        
    if label_arr is None:
        cLblFile = os.path.join(cDir,label_fname)
        if not os.path.exists(cLblFile):
            return {}
        label_img = nib.load(cLblFile)
        label_arr = label_img.get_data().astype(np.float64)
    
    mask = (label_arr >= 1) * 1
    
    mask_ct = np.sum(mask,axis=(0,1))
    slice_vec = np.argwhere(mask_ct >= min_size).flatten()
    
    n_slices = len(slice_vec)
    diag_dict =dict()
    for modal in modal_list:
        diag_dict[modal] = []
        if len(img[modal]) == 0:
            continue
        # [n_slices, x_dim, y_dim, views]
        dim = img[modal].shape
        if len(dim)>=3:
            n_slices = len(slice_vec)
        else:
            n_slices = 1

        for slI in range(n_slices):
            slice_ind = slice_vec[slI]

            # for view_ind in range(len(views)):
                # pick slice and view
            mask_orig = mask[:,:,slice_ind] == 1
            if np.sum(mask_orig,axis=(0,1)) < min_size:
                diag_dict[modal].append(list([]))  # should be caught earlier
#                 print(modal,slI,diag_dict[modal])
                continue
            # crop the image
            B = np.argwhere(mask_orig)
            (xstart_x, ystart_x), (xstop_x, ystop_x) = B.min(0), B.max(0) + 1  

            if (xstop_x-xstart_x <= 1) or (ystop_x-ystart_x <= 1):
#                 print(modal,slI,diag_dict[modal])
                continue 
            #convert numpy array into sitk
            img_slice = img[modal][xstart_x:xstop_x, ystart_x:ystop_x,slice_ind]
            mask_slice = mask[xstart_x:xstop_x, ystart_x:ystop_x,slice_ind]
            dim_rs = img_slice.shape

            slice_copy = deepcopy(img_slice)
            # outside of ROI = non_roi_val value
            slice_copy[np.where(mask_slice ==0)] = non_roi_val

            # flip upside down to match GUDHI PERSIUS data structure
            flipped = np.flipud(slice_copy)
            flattened = flipped.reshape(-1)
            cell_list = flattened.tolist() 
            
            # compute persistence
            CC_instance = CC(dimensions= [dim_rs[0], dim_rs[1]], top_dimensional_cells = cell_list)
            diag = CC_instance.persistence()
#             print(diag)
            # convert diag into numpy array for later analysis
            filtered = list()

            for elmt_ind in range(len(diag)):
                elmt = diag[elmt_ind]
                if elmt[1][1] != non_roi_val:
                    if elmt[1][1] == math.inf:
                        filtered.append( [elmt[0], elmt[1][0], lim] )
                    else:
                        filtered.append( [elmt[0], elmt[1][0], elmt[1][1]] )
            filtered = np.reshape(filtered,[-1,3])
            
            

            diag_dict[modal].append(filtered)
    return diag_dict

def get_tda_feats(cDir, modal_list = default_modal_list(), modal_dict = default_modal_dict(),label_fname = default_label_fname(), label_arr = None, feat_suffix = '', min_size = default_min_size(), non_roi_val = default_non_roi_val(), lim = default_lim()):
    
    diag_dict = get_diag_dict(cDir,modal_list = modal_list, modal_dict = modal_dict, label_fname = label_fname, label_arr = label_arr, min_size=min_size,non_roi_val = non_roi_val, lim = lim)
    
    (feature_names,feature_names_level) = get_feature_names(modal_list, suffix = feat_suffix)
    fparts = default_feat_parts()
    n_fparts = len(fparts)
    n_feats_per_modal = len(feature_names_level) 
    n_data = max([len(diag_dict[modal]) for modal in modal_list]) # pad out the number of slices to the maximum number of slices available
    feature_mat = list()
    for modal in modal_list:
        feature_modal = np.empty([n_data, n_feats_per_modal])
        feature_modal[:] = np.nan # fill slices with nan where unavailable

        for image_ind in range(len(diag_dict[modal])):
            feature_vec = list()
            diag = diag_dict[modal][image_ind]
#             if np.sum(mask[:,:,image_ind],axis=(0,1)) == 0:
            if len(diag) == 0:
                continue
            
            diag_list =[diag[np.where(diag[:,0]==0)], diag[np.where(diag[:,0]==1)]]

            for diag_elmt in diag_list:
                if len(diag_elmt) ==  0: # fill with zeros when the diagonal element doesn't exist
                    feature_vec.append([0]*n_fparts)
                else:
                    feature_vec.append( [f_1(diag_elmt), f_2(diag_elmt), f_3(diag_elmt), f_4(diag_elmt),
                                g_1(diag_elmt), g_2(diag_elmt), g_3(diag_elmt), g_4(diag_elmt),
                                g_5(diag_elmt), g_6(diag_elmt), g_7(diag_elmt), g_8(diag_elmt),
                                g_9(diag_elmt), g_10(diag_elmt), g_11(diag_elmt)] )

            flat_feature_vec = [item for sublist in feature_vec for item in sublist]
            feature_modal[image_ind] = flat_feature_vec

        feature_mat.append(feature_modal)

    features = np.hstack([mat for mat in feature_mat])
    return features,feature_names