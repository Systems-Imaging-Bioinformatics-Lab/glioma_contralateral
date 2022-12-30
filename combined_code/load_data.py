import numpy as np
import pandas as pd
import os
import file_structure

def load_data(dataset, fold_num, mode,ICC_filt=0,ICC_thresh=.75):
    (origdir,basedir,imagedir,normdir,splitdir,out_dir,model_dir) = file_structure.file_dirs(mode)
    
    if ICC_filt == 4:
        if ICC_thresh == .75:
            ICC_thresh = .9
        ICC_filt = 3
    if ICC_filt == 1:
        suffix = '_icc2'
    elif ICC_filt >= 3:
        suffix = '_icc%i' % ICC_filt
    else: 
        suffix = '' 
    
    fold_dir = os.path.join(splitdir,'fold_%i' % fold_num)
    
    # load pca
    pca_name = os.path.join(fold_dir, '%s_%s_pca%s.npy' % (dataset,mode[0:3].lower(),suffix))
    if os.path.isfile(pca_name):
        out_pca = np.load(pca_name)
    else:
        out_pca = []
    
    if mode == 'cmb': # data combined mode
        (tex_samples, _, out_1p19q, out_age, out_KPS, out_gender, out_id, 
         tex_names) = load_data(dataset, fold_num, 'texture',ICC_filt)
        (tda_samples, _, _, _, _, _, _, tda_names) = load_data(dataset, fold_num, 'TDA',ICC_filt)
        (cnn_samples, _, _, _, _, _, _, cnn_names) = load_data(dataset, fold_num, 'CNN',ICC_filt)
        out_samples = np.concatenate((tex_samples,tda_samples,cnn_samples),axis=1)
        feature_names = np.concatenate((tex_names,tda_names,cnn_names),axis=0)
        return out_samples, out_pca, out_1p19q, out_age, out_KPS, out_gender, out_id, feature_names
    if mode not in ('texture','TDA','CNN'):
        raise ValueError('Mode provided was not one of: (\'texture\',\'TDA\',\'CNN\')')
    
    # load train data
    os.chdir(splitdir)
    
    # load up per patient factors
    label_1p19q = np.load('label_1p19q.npy')
    label_1p19q = label_1p19q.astype(np.float32).reshape(-1,1)
    label_age = np.expand_dims(np.load('label_age.npy'),1)
    label_KPS = np.expand_dims(np.load('label_KPS.npy'),1)
    label_gender = np.expand_dims(np.load('label_gender.npy'),1)
    label_id = np.expand_dims(np.load('label_id.npy'),1)
    
    # load up features
    
    if dataset == 'all':
        setIdxs = np.arange(len(label_1p19q))
    else:
        setIdxs = np.load(os.path.join(fold_dir,'%s_idxs.npy' % dataset)) # e.g. train, test, val
    
    samples = np.load('slices_%s_features.npy' % mode)
    out_samples = np.empty(np.append([len(setIdxs)],samples.shape[1]))
    (out_1p19q,out_age,out_KPS,out_gender,out_id) = (np.empty(len(setIdxs)) for i in range(5))
    
    feature_names = np.load('%s_feature_names.npy' % mode)
    
    for s_o in range(len(setIdxs)):
        s_i = setIdxs[s_o]
        out_samples[s_o,:]  = samples[s_i,:,]

        out_1p19q[s_o]  = label_1p19q[s_i]
        out_age[s_o]    = label_age[s_i]
        out_KPS[s_o]    = label_KPS[s_i]
        out_gender[s_o] = label_gender[s_i]
        out_id[s_o]     = label_id[s_i]
     
    if ICC_filt >= 1:
        features_ICC = np.load(os.path.join(fold_dir,'%s_feature_min%s.npy' % (mode[0:3].lower(),suffix)))
        idxs = np.argwhere(features_ICC > ICC_thresh).ravel()
        out_samples = out_samples[:,idxs]
        feature_names = [feature_names[i] for i in idxs]
    os.chdir(origdir)
    

    return out_samples, out_pca, out_1p19q, out_age, out_KPS, out_gender, out_id, feature_names