import os
from radiomics import featureextractor
import SimpleITK as sitk
import pkg_resources
import numpy as np
import nibabel as nib

from copy import deepcopy
from scan_handler import norm_modals, default_modal_list, default_modal_dict

# def default_modal_list():
#     return ['FLAIR','T1', 'T1post','T2']
# def default_modal_dict():
#     return {'t1': 't1.nii.gz', 't1post':'t1Gd.nii.gz', 't2':'t2.nii.gz', 'flair': 'flair.nii.gz'}

def load_extractor(param_path = 'texture_settings.yaml'):
    #load the parameters for extractor
    # param_path = os.path.join(origdir,'texture_settings.yaml')
    pyrad_ver = pkg_resources.parse_version(pkg_resources.get_distribution("pyradiomics").version)
    if pyrad_ver >= pkg_resources.parse_version("2.1.2"):
        # changed the name of the feature extractor some time between 2.1.0 and 2.1.2
        extractor = featureextractor.RadiomicsFeatureExtractor(param_path)
    else:
        extractor = featureextractor.RadiomicsFeaturesExtractor(param_path)
    return extractor




def get_tex_feats(cDir,label_arr = None, label_fname = 'truth.nii.gz', feat_suffix = '', min_size = 10,modal_list = default_modal_list(), modal_dict = default_modal_dict()):
    # input is directory, output is array of features
    
    img = norm_modals(cDir,modal_list = modal_list, modal_dict = modal_dict)
    extractor = load_extractor()
    
    cLblFile = os.path.join(cDir,label_fname)
    if not os.path.exists(cLblFile):
        return np.array([]), []
    label_img = nib.load(cLblFile)
    
    if label_arr is None:
        label_arr = label_img.get_data().astype(np.float64)

    data_spacing = label_img.header['pixdim'][1:4]
    mask = (label_arr >= 1) * 1
    
    mask_ct = np.sum(mask,axis=(0,1))
    slice_vec = np.argwhere(mask_ct >= min_size).flatten()
    n_slices = len(slice_vec)

    features = {}
    for modal in modal_list:
        features[modal] = {}
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
            features[modal][slI] = {}

            # for view_ind in range(len(views)):
                # pick slice and view
            mask_orig = mask[:,:,slice_ind] == 1
            if np.sum(mask_orig,axis=(0,1)) < min_size:
                features[modal][slI]['z']=[] # should be caught earlier
                continue
            # crop the image
            B = np.argwhere(mask_orig)
            (xstart_x, ystart_x), (xstop_x, ystop_x) = B.min(0), B.max(0) + 1  

            if (xstop_x-xstart_x <= 1) or (ystop_x-ystart_x <= 1):
                features[modal][slI]['z']=[]
                continue 
            #convert numpy array into sitk
            img_slice = img[modal][xstart_x:xstop_x, ystart_x:ystop_x,slice_ind]
            mask_slice = mask[xstart_x:xstop_x, ystart_x:ystop_x,slice_ind]

            sitk_img = sitk.GetImageFromArray(img_slice) 
            sitk_img.SetSpacing((float(data_spacing[0]), float(data_spacing[1]), float(data_spacing[2]) ))
            sitk_img = sitk.JoinSeries(sitk_img)

            sitk_mask = sitk.GetImageFromArray(mask_slice)
            sitk_mask.SetSpacing((float(data_spacing[0]), float(data_spacing[1]), float(data_spacing[2]) ))
            sitk_mask = sitk.JoinSeries(sitk_mask)
            sitk_mask = sitk.Cast(sitk_mask, sitk.sitkInt32)
            try:
                features[modal][slI]['z']=extractor.execute(sitk_img, sitk_mask)
            except Exception as e:
                print('Exception: ',e) # exception of last resort
                features[modal][slI]['z']= []

    # obtain feature names
    for i in range(len( features[modal_list[0]])):
        featList =  features[modal_list[0]][i]['z']
        if len(featList)>0:
            break
    feature_names = list(sorted(filter ( lambda k: k.startswith("original_"),featList)))

    feature_names_all = []
    for modal in modal_list:
        feature_names_all.append([elemt + '_'+ modal + feat_suffix for elemt in feature_names])

    feature_names_all_flattened = [val for sublist in feature_names_all for val in sublist]

    # remove "original_"
    feature_names_all_flattened = [elmt[9:] for elmt in feature_names_all_flattened]

#     print('number of %s features = %i' % (modal, len(feature_names_all_flattened)))

    #Make a numpy array of all the values
    samples = np.zeros( (n_slices, len(feature_names)*len(modal_list) ) )
    for slI in range(n_slices):
        a = np.array([])

        for modal in modal_list:

            for feature_name in feature_names:
                try:
                    if features[modal][slI]['z']:
                        newFeat = features[modal][slI]['z'][feature_name]
                    else:
                        newFeat = np.nan
                except KeyError:
                    #print ('I got a KeyError - %s' % str(feature_name))
                    newFeat = np.nan
                a = np.append(a, newFeat)
#         print(a.shape)
        samples[slI,:] = a
    return samples, feature_names_all_flattened
