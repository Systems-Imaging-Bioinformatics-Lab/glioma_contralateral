import os
import nibabel as nib
import numpy as np
# import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from copy import deepcopy
from nipy import labs
import multiprocessing
from joblib import Parallel, delayed
from scipy.stats import iqr

basedir = '/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/data/image_data/'
os.chdir(basedir)
patients=next(os.walk('.'))[1]

normalized_dir = '/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/data/normalized_data/'

def normalize(p):

    print(p, patients[p])
    patient_dir = basedir + patients[p] + '/'
    patient_norm_dir = normalized_dir + patients[p] + '/'
    os.chdir(patient_dir)

    FLAIR_ss = np.round(nib.load('flair.nii.gz').get_data())
    T2_ss = np.round(nib.load('t2.nii.gz').get_data())
    T1_ss = np.round(nib.load('t1.nii.gz').get_data())
    T1post_ss = np.round(nib.load('t1Gd.nii.gz').get_data())
    truth = nib.load('truth.nii.gz').get_data()
    
    #normalize image intensity values relative to normal brain
    idx_mask = np.where(truth==0)
    idx_nz = np.nonzero(FLAIR_ss[idx_mask])
    median = np.median(FLAIR_ss[idx_mask][idx_nz])
    curr_iqr = iqr(FLAIR_ss[idx_mask][idx_nz])
    FLAIR_normssn4 = deepcopy(FLAIR_ss)
    FLAIR_normssn4[np.nonzero(FLAIR_ss)] = (FLAIR_normssn4[np.nonzero(FLAIR_ss)]-median)/curr_iqr
    idx_nz = np.nonzero(T2_ss[idx_mask])
    median = np.median(T2_ss[idx_mask][idx_nz])
    curr_iqr = iqr(T2_ss[idx_mask][idx_nz])
    T2_normssn4 = deepcopy(T2_ss)
    T2_normssn4[np.nonzero(T2_ss)] = (T2_normssn4[np.nonzero(T2_ss)]-median)/curr_iqr
    idx_nz = np.nonzero(T1_ss[idx_mask])
    median = np.median(T1_ss[idx_mask][idx_nz])
    curr_iqr = iqr(T1_ss[idx_mask][idx_nz])
    T1_normssn4 = deepcopy(T1_ss)
    T1_normssn4[np.nonzero(T1_ss)] = (T1_normssn4[np.nonzero(T1_ss)]-median)/curr_iqr   
    idx_nz = np.nonzero(T1post_ss[idx_mask])
    median = np.median(T1post_ss[idx_mask][idx_nz])
    curr_iqr = iqr(T1post_ss[idx_mask][idx_nz])
    T1post_normssn4 = deepcopy(T1post_ss)
    T1post_normssn4[np.nonzero(T1post_ss)] = (T1post_normssn4[np.nonzero(T1post_ss)]-median)/curr_iqr              
                
    if not os.path.exists(patient_norm_dir):
        os.makedirs(patient_norm_dir)

    os.chdir(patient_norm_dir)
    np.save('FLAIR_normssn4.npy',FLAIR_normssn4)
    np.save('T2_normssn4.npy',T2_normssn4)
    np.save('T1_normssn4.npy',T1_normssn4)
    np.save('T1post_normssn4.npy',T1post_normssn4)
    np.save('truth.npy', truth)

num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(delayed(normalize)(p) for p in range(len(patients)))