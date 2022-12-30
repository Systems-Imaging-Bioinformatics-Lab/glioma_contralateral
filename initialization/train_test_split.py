from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import os

# load data and labels
os.chdir('/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/data/pre_processed_data_1p_19q_balanced')
slices_FLAIR = np.load('slices_FLAIR.npy')
slices_T2 = np.load('slices_T2.npy')
slices_T1 = np.load('slices_T1.npy')
slices_T1post = np.load('slices_T1post.npy')

label_1p19q = np.load('label_1p19q.npy')
label_age = np.load('label_age.npy')
label_KPS = np.load('label_KPS.npy')
label_gender = np.load('label_gender.npy')
label_IDH1 = np.load('label_IDH1.npy')
label_OS = np.load('label_OS.npy')

#Split the data in 70/10/20 manner among train/val/test

# set random seed
random_seed = 42

sss = StratifiedShuffleSplit(n_splits =1, test_size = 0.2, random_state = random_seed)

# split between train and test
for train_idx, test_idx in sss.split(slices_FLAIR, label_1p19q):
	train_FLAIR, test_FLAIR = slices_FLAIR[train_idx], slices_FLAIR[test_idx]
	train_T2, test_T2 = slices_T2[train_idx], slices_T2[test_idx]
	train_T1, test_T1 = slices_T1[train_idx], slices_T1[test_idx]
	train_T1post, test_T1post = slices_T1post[train_idx], slices_T1post[test_idx]

	train_1p19q, test_1p19q = label_1p19q[train_idx], label_1p19q[test_idx]
	train_age, test_age = label_age[train_idx], label_age[test_idx]
	train_KPS, test_KPS = label_KPS[train_idx], label_KPS[test_idx]
	train_gender, test_gender = label_gender[train_idx], label_gender[test_idx]	
	train_OS, test_OS = label_OS[train_idx], label_OS[test_idx]
	train_IDH1, test_IDH1 = label_IDH1[train_idx], label_IDH1[test_idx]

# split between train and val

random_seed2 = 32

sss2 = StratifiedShuffleSplit(n_splits =1, test_size = 0.125)

for train_only_idx, val_idx in sss2.split(train_FLAIR, train_1p19q):
	train_only_FLAIR, val_FLAIR = train_FLAIR[train_only_idx], train_FLAIR[val_idx]
	train_only_T2, val_T2 = train_T2[train_only_idx], train_T2[val_idx]
	train_only_T1, val_T1 = train_T1[train_only_idx], train_T1[val_idx]
	train_only_T1post, val_T1post = train_T1post[train_only_idx], train_T1post[val_idx]

	train_only_1p19q, val_1p19q = train_1p19q[train_only_idx], train_1p19q[val_idx]
	train_only_age, val_age = train_age[train_only_idx], train_age[val_idx]
	train_only_KPS, val_KPS = train_KPS[train_only_idx], train_KPS[val_idx]
	train_only_gender, val_gender = train_gender[train_only_idx], train_gender[val_idx]
	train_only_OS, val_OS = train_OS[train_only_idx], train_OS[val_idx]
	train_only_IDH1, val_IDH1 = train_IDH1[train_only_idx], train_IDH1[val_idx]

save_dir = '/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/data/data_splitted_balanced_20/'

train_dir = save_dir +'train/'
val_dir = save_dir + 'val/'
test_dir = save_dir + 'test/'

os.chdir(train_dir)
np.save('train_FLAIR.npy', train_only_FLAIR)
np.save('train_T2.npy', train_only_T2)
np.save('train_T1.npy', train_only_T1)
np.save('train_T1post.npy', train_only_T1post)

np.save('train_age.npy',train_only_age)
np.save('train_IDH1.npy', train_only_IDH1)
np.save('train_1p19q.npy', train_only_1p19q)
np.save('train_OS.npy', train_only_OS)
np.save('train_gender.npy', train_only_gender)
np.save('train_KPS.npy', train_only_KPS)

os.chdir(val_dir)
np.save('val_FLAIR.npy', val_FLAIR)
np.save('val_T2.npy', val_T2)
np.save('val_T1.npy', val_T1)
np.save('val_T1post.npy', val_T1post)

np.save('val_age.npy',val_age)
np.save('val_IDH1.npy', val_IDH1)
np.save('val_1p19q.npy', val_1p19q)
np.save('val_OS.npy', val_OS)
np.save('val_gender.npy', val_gender)
np.save('val_KPS.npy', val_KPS)

os.chdir(test_dir)
np.save('test_FLAIR.npy', test_FLAIR)
np.save('test_T2.npy', test_T2)
np.save('test_T1.npy', test_T1)
np.save('test_T1post.npy', test_T1post)

np.save('test_age.npy',test_age)
np.save('test_IDH1.npy', test_IDH1)
np.save('test_1p19q.npy', test_1p19q)
np.save('test_OS.npy', test_OS)
np.save('test_gender.npy', test_gender)
np.save('test_KPS.npy', test_KPS)