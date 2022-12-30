import os
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn import preprocessing

# obtain top 50 percentile images

#Load a spreadsheet with patient ID in the first column, age in the third column, and IDH status in the second column
xl = pd.ExcelFile('/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/data/clinical_data_IDH1_1p19q.xlsx')
df = np.asarray(xl.parse("KPS_80"))

patient_age = df[:,[0,1]]
patient_IDH = df[:,[0,2]]
patient_1p19q = df[:,[0,3]]
patient_IDH1_1p19q = df[:,[0,4]]
patient_OS = df[:,[0,5]]
patient_gender = df[:,[0,6]]
patient_KPS = df[:,[0,7]]

basedir = '/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/data/normalized_data/'
os.chdir(basedir)
patients=next(os.walk('.'))[1]

desired_size=[142,142]

# obtain the ratio between codel and non_codel
num_codel = np.count_nonzero(patient_1p19q[:,1])
num_non_codel = len(patient_1p19q)- num_codel
#codel: non_codel = 13:130 = 1:10

# for balancing purpose between codel vs non-codel cases, extract 30 codel per 3 non-codel case => 50:50

# pre allocate arrays for codel and non codel separately

# case: non-codel 130*3 = 390

non_codel_factor = 2

slices_FLAIR = np.empty([num_non_codel*non_codel_factor, desired_size[0], desired_size[1], 3])
slices_T2 = np.empty([num_non_codel*non_codel_factor, desired_size[0], desired_size[1], 3])
slices_T1 = np.empty([num_non_codel*non_codel_factor, desired_size[0], desired_size[1], 3])
slices_T1post = np.empty([num_non_codel*non_codel_factor, desired_size[0], desired_size[1], 3])

label_age = np.empty(num_non_codel*non_codel_factor)
label_IDH1 = np.empty(num_non_codel*non_codel_factor)
label_1p19q = np.empty(num_non_codel*non_codel_factor)
label_IDH1_1p19q = np.empty(num_non_codel*non_codel_factor)
label_OS = np.empty(num_non_codel*non_codel_factor)
label_gender = np.empty(num_non_codel*non_codel_factor)
label_KPS = np.empty(num_non_codel*non_codel_factor) 

# case: codel 13*30 = 390

codel_factor = 20

slices_FLAIR_codel = np.empty([num_codel*codel_factor, desired_size[0], desired_size[1], 3])
slices_T2_codel = np.empty([num_codel*codel_factor, desired_size[0], desired_size[1], 3])
slices_T1_codel = np.empty([num_codel*codel_factor, desired_size[0], desired_size[1], 3])
slices_T1post_codel = np.empty([num_codel*codel_factor, desired_size[0], desired_size[1], 3])

label_age_codel = np.empty(num_codel*codel_factor)
label_IDH1_codel = np.empty(num_codel*codel_factor)
label_1p19q_codel = np.empty(num_codel*codel_factor)
label_IDH1_1p19q_codel = np.empty(num_codel*codel_factor)
label_OS_codel = np.empty(num_codel*codel_factor)
label_gender_codel = np.empty(num_codel*codel_factor)
label_KPS_codel = np.empty(num_codel*codel_factor) 

def zoompad(array, desired_size):
	array = cv2.resize(array,(desired_size[0],desired_size[1]))
	return array
	
#counters
codel_counter = 0
non_codel_counter = 0
for p in range(len(patients)):
	
	print(p, patients[p])
	patient_dir = basedir + patients[p] + '/'

	idx_idh1=np.asarray(np.where((patient_IDH[:,0].astype(str))==str(patients[p])))

	curr_idh1 = patient_IDH[idx_idh1,1]

	idx_age=np.asarray(np.where((patient_age[:,0].astype(str))==str(patients[p])))
	curr_age = patient_age[idx_age,1]

	idx_1p19q=np.asarray(np.where((patient_1p19q[:,0].astype(str))==str(patients[p])))
	curr_1p19q = patient_1p19q[idx_1p19q,1]

	idx_IDH1_1p19q=np.asarray(np.where((patient_IDH1_1p19q[:,0].astype(str))==str(patients[p])))
	curr_IDH1_1p19q = patient_IDH1_1p19q[idx_IDH1_1p19q,1]

	idx_OS=np.asarray(np.where((patient_OS[:,0].astype(str))==str(patients[p])))
	curr_OS = patient_OS[idx_OS,1]

	idx_gender=np.asarray(np.where((patient_gender[:,0].astype(str))==str(patients[p])))
	curr_gender=patient_gender[idx_gender,1]

	idx_KPS=np.asarray(np.where((patient_KPS[:,0].astype(str))==str(patients[p])))
	curr_KPS=patient_KPS[idx_KPS,1]

	os.chdir(patient_dir)
	FLAIR = np.load('FLAIR_normssn4.npy')
	T2 = np.load('T2_normssn4.npy')
	T1 = np.load('T1_normssn4.npy')
	T1post = np.load('T1post_normssn4.npy')
	mask = np.load('truth.npy')

	# FLAIR = nib.load('flair.nii.gz').get_data()
	# T2 = nib.load('t2.nii.gz').get_data()
	# T1 = nib.load('t1.nii.gz').get_data()
	# T1post = nib.load('t1Gd.nii.gz').get_data()
	# mask = nib.load('truth.nii.gz').get_data()

	# mask label is organized as following: 1 = non-enhancing, 2 = edema, 4 = enhancing 
	mask[mask==2] = 1
	mask[mask==4] = 1

	FLAIR_m= FLAIR
	T2_m= T2
	T1_m= T1
	T1post_m= T1post
	
	#Find the largest, 75th, and 50th percentile slices in each dimension
	x_sum=np.sum(mask,axis=(1,2))
	y_sum=np.sum(mask,axis=(0,2))
	z_sum=np.sum(mask,axis=(0,1))

	#Check if the patient is codel or non-codel (1 or 0). The ratio between codel and non-codel is 13:130
	#So, subsample codel cases 10 times more than codel

	if curr_1p19q == 0: #i.e. if non-codel case, subsample 100,75, and 50 percentile

		xp100=np.percentile(x_sum[np.nonzero(x_sum)],100,interpolation='nearest')
		xp75=np.percentile(x_sum[np.nonzero(x_sum)],75,interpolation='nearest')
		# xp50=np.percentile(x_sum[np.nonzero(x_sum)],50,interpolation='nearest')
		yp100=np.percentile(y_sum[np.nonzero(y_sum)],100,interpolation='nearest')
		yp75=np.percentile(y_sum[np.nonzero(y_sum)],75,interpolation='nearest')
		# yp50=np.percentile(y_sum[np.nonzero(y_sum)],50,interpolation='nearest')
		zp100=np.percentile(z_sum[np.nonzero(z_sum)],100,interpolation='nearest')
		zp75=np.percentile(z_sum[np.nonzero(z_sum)],75,interpolation='nearest')
		# zp50=np.percentile(z_sum[np.nonzero(z_sum)],50,interpolation='nearest')
		
		x_idx = np.argwhere(x_sum==xp100)[0][0]
		y_idx = np.argwhere(y_sum==yp100)[0][0]
		z_idx = np.argwhere(z_sum==zp100)[0][0]
		
		B = np.argwhere(mask[x_idx])
		(xstart_x, ystart_x), (xstop_x, ystop_x) = B.min(0), B.max(0) + 1     
		B = np.argwhere(mask[:,y_idx])
		(xstart_y, ystart_y), (xstop_y, ystop_y) = B.min(0), B.max(0) + 1     
		B = np.argwhere(mask[:,:,z_idx])
		(xstart_z, ystart_z), (xstop_z, ystop_z) = B.min(0), B.max(0) + 1
		
		FLAIR_x1 = zoompad(np.asarray(FLAIR_m[x_idx][xstart_x:xstop_x, ystart_x:ystop_x]), desired_size)
		FLAIR_y1 = zoompad(np.asarray(FLAIR_m[:,y_idx][xstart_y:xstop_y, ystart_y:ystop_y]), desired_size)
		FLAIR_z1 = zoompad(np.asarray(FLAIR_m[:,:,z_idx][xstart_z:xstop_z, ystart_z:ystop_z]), desired_size)
		
		T2_x1 = zoompad(np.asarray(T2_m[x_idx][xstart_x:xstop_x, ystart_x:ystop_x]), desired_size)
		T2_y1 = zoompad(np.asarray(T2_m[:,y_idx][xstart_y:xstop_y, ystart_y:ystop_y]), desired_size)
		T2_z1 = zoompad(np.asarray(T2_m[:,:,z_idx][xstart_z:xstop_z, ystart_z:ystop_z]), desired_size)
		
		T1_x1 = zoompad(np.asarray(T1_m[x_idx][xstart_x:xstop_x, ystart_x:ystop_x]), desired_size)
		T1_y1 = zoompad(np.asarray(T1_m[:,y_idx][xstart_y:xstop_y, ystart_y:ystop_y]), desired_size)
		T1_z1 = zoompad(np.asarray(T1_m[:,:,z_idx][xstart_z:xstop_z, ystart_z:ystop_z]), desired_size)
		
		T1post_x1 = zoompad(np.asarray(T1post_m[x_idx][xstart_x:xstop_x, ystart_x:ystop_x]), desired_size)
		T1post_y1 = zoompad(np.asarray(T1post_m[:,y_idx][xstart_y:xstop_y, ystart_y:ystop_y]), desired_size)
		T1post_z1 = zoompad(np.asarray(T1post_m[:,:,z_idx][xstart_z:xstop_z, ystart_z:ystop_z]), desired_size)
		
		x_idx = np.argwhere(x_sum==xp75)[0][0]
		y_idx = np.argwhere(y_sum==yp75)[0][0]
		z_idx = np.argwhere(z_sum==zp75)[0][0]
		
		B = np.argwhere(mask[x_idx])
		(xstart_x, ystart_x), (xstop_x, ystop_x) = B.min(0), B.max(0) + 1     
		B = np.argwhere(mask[:,y_idx])
		(xstart_y, ystart_y), (xstop_y, ystop_y) = B.min(0), B.max(0) + 1     
		B = np.argwhere(mask[:,:,z_idx])
		(xstart_z, ystart_z), (xstop_z, ystop_z) = B.min(0), B.max(0) + 1
		
		FLAIR_x2 = zoompad(np.asarray(FLAIR_m[x_idx][xstart_x:xstop_x, ystart_x:ystop_x]), desired_size)
		FLAIR_y2 = zoompad(np.asarray(FLAIR_m[:,y_idx][xstart_y:xstop_y, ystart_y:ystop_y]), desired_size)
		FLAIR_z2 = zoompad(np.asarray(FLAIR_m[:,:,z_idx][xstart_z:xstop_z, ystart_z:ystop_z]), desired_size)
		
		T2_x2 = zoompad(np.asarray(T2_m[x_idx][xstart_x:xstop_x, ystart_x:ystop_x]), desired_size)
		T2_y2 = zoompad(np.asarray(T2_m[:,y_idx][xstart_y:xstop_y, ystart_y:ystop_y]), desired_size)
		T2_z2 = zoompad(np.asarray(T2_m[:,:,z_idx][xstart_z:xstop_z, ystart_z:ystop_z]), desired_size)
		
		T1_x2 = zoompad(np.asarray(T1_m[x_idx][xstart_x:xstop_x, ystart_x:ystop_x]), desired_size)
		T1_y2 = zoompad(np.asarray(T1_m[:,y_idx][xstart_y:xstop_y, ystart_y:ystop_y]), desired_size)
		T1_z2 = zoompad(np.asarray(T1_m[:,:,z_idx][xstart_z:xstop_z, ystart_z:ystop_z]), desired_size)
		
		T1post_x2 = zoompad(np.asarray(T1post_m[x_idx][xstart_x:xstop_x, ystart_x:ystop_x]), desired_size)
		T1post_y2 = zoompad(np.asarray(T1post_m[:,y_idx][xstart_y:xstop_y, ystart_y:ystop_y]), desired_size)
		T1post_z2 = zoompad(np.asarray(T1post_m[:,:,z_idx][xstart_z:xstop_z, ystart_z:ystop_z]), desired_size)
		
		# save in ascending order
		slices_FLAIR[non_codel_factor*non_codel_counter] = np.stack((FLAIR_x2, FLAIR_y2, FLAIR_z2), axis=2)
		slices_FLAIR[non_codel_factor*non_codel_counter+1] = np.stack((FLAIR_x1, FLAIR_y1, FLAIR_z1), axis=2)

		slices_T2[non_codel_factor*non_codel_counter] = np.stack((T2_x2, T2_y2, T2_z2), axis=2)
		slices_T2[non_codel_factor*non_codel_counter+1] = np.stack((T2_x1, T2_y1, T2_z1), axis=2)

		slices_T1[non_codel_factor*non_codel_counter] = np.stack((T1_x2, T1_y2, T1_z2), axis=2)
		slices_T1[non_codel_factor*non_codel_counter+1] = np.stack((T1_x1, T1_y1, T1_z1), axis=2)

		slices_T1post[non_codel_factor*non_codel_counter] = np.stack((T1post_y2, T1post_y2, T1post_z2), axis=2)
		slices_T1post[non_codel_factor*non_codel_counter+1] = np.stack((T1post_x1, T1post_y1, T1post_z1), axis=2)

		label_age[non_codel_factor*non_codel_counter:non_codel_factor*non_codel_counter+non_codel_factor] = curr_age
		label_IDH1[non_codel_factor*non_codel_counter:non_codel_factor*non_codel_counter+non_codel_factor] = curr_idh1
		label_1p19q[non_codel_factor*non_codel_counter:non_codel_factor*non_codel_counter+non_codel_factor] = curr_1p19q
		label_IDH1_1p19q[non_codel_factor*non_codel_counter:non_codel_factor*non_codel_counter+non_codel_factor] = curr_IDH1_1p19q
		label_OS[non_codel_factor*non_codel_counter:non_codel_factor*non_codel_counter+non_codel_factor] = curr_OS
		label_gender[non_codel_factor*non_codel_counter:non_codel_factor*non_codel_counter+non_codel_factor] = curr_gender
		label_KPS[non_codel_factor*non_codel_counter:non_codel_factor*non_codel_counter+non_codel_factor] = curr_KPS

		non_codel_counter += 1

	elif curr_1p19q == 1:
		#obtain top 30 mask area indices
		x_ind_array = np.argpartition(x_sum,-codel_factor)[-codel_factor:]
		x_ind_array = x_ind_array[np.argsort(x_sum[x_ind_array])]
		
		y_ind_array = np.argpartition(y_sum,-codel_factor)[-codel_factor:]
		y_ind_array = y_ind_array[np.argsort(y_sum[y_ind_array])]
		
		z_ind_array = np.argpartition(z_sum,-codel_factor)[-codel_factor:]
		z_ind_array = z_ind_array[np.argsort(z_sum[z_ind_array])]

		for slice_ind in range(codel_factor): # store in ascending order

			x_idx = x_ind_array[slice_ind]
			y_idx = y_ind_array[slice_ind]
			z_idx = z_ind_array[slice_ind]
		
			B = np.argwhere(mask[x_idx]) # y-z plane
			(xstart_x, ystart_x), (xstop_x, ystop_x) = B.min(0), B.max(0) + 1 
			B = np.argwhere(mask[:,y_idx]) # x-z plane
			(xstart_y, ystart_y), (xstop_y, ystop_y) = B.min(0), B.max(0) + 1 
			B = np.argwhere(mask[:,:,z_idx]) # x-y plane
			(xstart_z, ystart_z), (xstop_z, ystop_z) = B.min(0), B.max(0) + 1
		
			FLAIR_x = zoompad(np.asarray(FLAIR_m[x_idx][xstart_x:xstop_x, ystart_x:ystop_x]), desired_size)
			FLAIR_y = zoompad(np.asarray(FLAIR_m[:,y_idx][xstart_y:xstop_y, ystart_y:ystop_y]), desired_size)
			FLAIR_z = zoompad(np.asarray(FLAIR_m[:,:,z_idx][xstart_z:xstop_z, ystart_z:ystop_z]), desired_size)
			
			T2_x = zoompad(np.asarray(T2_m[x_idx][xstart_x:xstop_x, ystart_x:ystop_x]), desired_size)
			T2_y = zoompad(np.asarray(T2_m[:,y_idx][xstart_y:xstop_y, ystart_y:ystop_y]), desired_size)
			T2_z = zoompad(np.asarray(T2_m[:,:,z_idx][xstart_z:xstop_z, ystart_z:ystop_z]), desired_size)
			
			T1_x = zoompad(np.asarray(T1_m[x_idx][xstart_x:xstop_x, ystart_x:ystop_x]), desired_size)
			T1_y = zoompad(np.asarray(T1_m[:,y_idx][xstart_y:xstop_y, ystart_y:ystop_y]), desired_size)
			T1_z = zoompad(np.asarray(T1_m[:,:,z_idx][xstart_z:xstop_z, ystart_z:ystop_z]), desired_size)
			
			T1post_x = zoompad(np.asarray(T1post_m[x_idx][xstart_x:xstop_x, ystart_x:ystop_x]), desired_size)
			T1post_y = zoompad(np.asarray(T1post_m[:,y_idx][xstart_y:xstop_y, ystart_y:ystop_y]), desired_size)
			T1post_z = zoompad(np.asarray(T1post_m[:,:,z_idx][xstart_z:xstop_z, ystart_z:ystop_z]), desired_size)

			slices_FLAIR_codel[codel_factor*codel_counter + slice_ind] = np.stack((FLAIR_x, FLAIR_y, FLAIR_z), axis=2)
			slices_T2_codel[codel_factor*codel_counter + slice_ind] = np.stack((T2_x, T2_y, T2_z), axis=2)
			slices_T1_codel[codel_factor*codel_counter + slice_ind] = np.stack((T1_x, T1_y, T1_z), axis=2)
			slices_T1post_codel[codel_factor*codel_counter + slice_ind] = np.stack((T1post_x, T1post_y, T1post_z), axis=2)

		print("Done saving 20 slices")


		label_age_codel[codel_factor*codel_counter:codel_factor*codel_counter+codel_factor] = curr_age
		label_IDH1_codel[codel_factor*codel_counter:codel_factor*codel_counter+codel_factor] = curr_idh1
		label_1p19q_codel[codel_factor*codel_counter:codel_factor*codel_counter+codel_factor] = curr_1p19q
		label_IDH1_1p19q_codel[codel_factor*codel_counter:codel_factor*codel_counter+codel_factor] = curr_IDH1_1p19q
		label_OS_codel[codel_factor*codel_counter:codel_factor*codel_counter+codel_factor] = curr_OS
		label_gender_codel[codel_factor*codel_counter:codel_factor*codel_counter+codel_factor] = curr_gender
		label_KPS_codel[codel_factor*codel_counter:codel_factor*codel_counter+codel_factor] = curr_KPS

		codel_counter += 1

	print("Done patient number: ", p)
	del FLAIR, T2, T1, T1post, mask, curr_age, curr_idh1, curr_1p19q, curr_IDH1_1p19q, curr_OS, curr_gender, curr_KPS


# Combine codel and non-codel together
slices_FLAIR_comb = np.vstack((slices_FLAIR,slices_FLAIR_codel))
slices_T1_comb = np.vstack((slices_T1,slices_T1_codel))
slices_T1post_comb = np.vstack((slices_T1post,slices_T1post_codel))
slices_T2_comb = np.vstack((slices_T2,slices_T2_codel))

label_age_comb = np.append(label_age, label_age_codel)
label_IDH1_comb = np.append(label_IDH1,label_IDH1_codel)
label_1p19q_comb = np.append(label_1p19q,label_1p19q_codel)
label_gender_comb = np.append(label_gender, label_gender_codel)
label_KPS_comb = np.append(label_KPS, label_KPS_codel)
label_OS_comb = np.append(label_OS, label_OS_codel)
label_IDH1_1p19q_comb = np.append(label_IDH1_1p19q, label_IDH1_1p19q_codel)
#specify save directory
#save all patient data into numpy files

save_dir = '/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/data/pre_processed_data_1p_19q_balanced_20'

os.chdir(save_dir)
np.save('slices_FLAIR.npy', slices_FLAIR_comb)
np.save('slices_T2.npy', slices_T2_comb)
np.save('slices_T1.npy', slices_T1_comb)
np.save('slices_T1post.npy', slices_T1post_comb)

np.save('label_age.npy',label_age_comb)
np.save('label_IDH1.npy', label_IDH1_comb)
np.save('label_1p19q.npy', label_1p19q_comb)
np.save('label_IDH1_1p19q.npy', label_IDH1_1p19q_comb)
np.save('label_OS.npy', label_OS_comb)
np.save('label_gender.npy', label_gender_comb)
np.save('label_KPS.npy', label_KPS_comb)