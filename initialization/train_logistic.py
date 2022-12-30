import os
import numpy as np
from keras.models import load_model
from sklearn.linear_model import LogisticRegression

def get_accuracy(all_gt, all_label):
    return len(np.argwhere(all_gt==all_label))/float(len(all_gt))

def get_sensitivity(all_gt, all_label):
    loc = np.where(all_gt==1)
    return len(np.argwhere(all_gt[loc]==all_label[loc]))/float(len(all_gt[loc]))

def get_specificity(all_gt, all_label):
    loc = np.where(all_gt==0)
    return len(np.argwhere(all_gt[loc]==all_label[loc]))/float(len(all_gt[loc]))

from sklearn.metrics import roc_auc_score
from sklearn import linear_model

def get_auc(y_true,y_pred):
    n_bootstraps = 1000
    bootstrapped_scores = []
    np.random.seed(seed)
    for i in range(n_bootstraps):
        indices = np.random.choice(range(0, len(y_pred)), len(y_pred), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    return roc_auc_score(y_true, y_pred), confidence_lower, confidence_upper

def hyper_parameter_search(train_feat, train_1p19q, val_feat, val_1p19q, C_array):
    result = {'accuracy': [], 'AUC': [], 'sensitivity': [], 'specificity': [], 'C_val': C_array}

    for C_val in C_array:

        logreg_l1 = LogisticRegression(C=C_val, penalty = 'l1', tol=0.01)
        logreg_l2 = LogisticRegression(C=C_val, penalty = 'l2', tol=0.01)
        
        logreg_l1.fit(train_sig_imaging, train_1p19q)
        logreg_l2.fit(train_sig_imaging, train_1p19q)

        Z_l1 = logreg_l1.predict_proba(val_sig_imaging)
        Z_l2 = logreg_l2.predict_proba(val_sig_imaging)

        # accuracy
        auc_l1 = get_auc(val_1p19q, Z_l1[:,1])[0]
        auc_l2 = get_auc(val_1p19q, Z_l2[:,1])[0]
        # AUC
        acc_l1 = get_accuracy(val_1p19q, np.round(Z_l1[:,1]))
        acc_l2 = get_accuracy(val_1p19q, np.round(Z_l1[:,1]))
        # sensitivity
        sen_l1 = get_sensitivity(val_1p19q, np.round(Z_l1[:,1]))
        sen_l2 = get_sensitivity(val_1p19q, np.round(Z_l1[:,1]))
        # specificity
        spe_l1 = get_specificity(val_1p19q, np.round(Z_l1[:,1]))
        spe_l2 = get_specificity(val_1p19q, np.round(Z_l1[:,1]))

        result['AUC'].append([auc_l1, auc_l2])
        result['accuracy'].append([acc_l1, acc_l2])
        result['sensitivity'].append([sen_l1, sen_l2])
        result['specificity'].append([spe_l2, spe_l2])

    # Now, obtain the best C-val with best AUC and Accuracy combo
    AUC_list = result['AUC']
    accuracy_list = result['accuracy']

    summed = np.add(AUC_list, accuracy_list)

    row,col = np.where(summed==summed.max())

    if col[0] == 0:
        best_penalty_type = 'l1'
    else:
        best_penalty_type = 'l2'

    best_c = C_array[row][0]


    return best_penalty_type, best_c

#Specify which GPU to be used
os.environ["CUDA_VISIBLE_DEVICES"]="0"

seed = 0

# load train data
os.chdir('/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/data/data_splitted_balanced_20/train')
train_FLAIR = np.load('train_FLAIR.npy')
train_T2 = np.load('train_T2.npy')
train_T1 = np.load('train_T1.npy')
train_T1post = np.load('train_T1post.npy')

train_1p19q = np.load('train_1p19q.npy')
train_1p19q = train_1p19q.astype(np.float32).reshape(-1,1)
train_age = np.expand_dims(np.load('train_age.npy'),1)
train_KPS = np.expand_dims(np.load('train_KPS.npy'),1)
train_gender = np.expand_dims(np.load('train_gender.npy'),1)

# load val data
os.chdir('/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/data/data_splitted_balanced_20/val')
val_FLAIR = np.load('val_FLAIR.npy')
val_T2 = np.load('val_T2.npy')
val_T1 = np.load('val_T1.npy')
val_T1post = np.load('val_T1post.npy')

val_1p19q = np.load('val_1p19q.npy')
val_1p19q = val_1p19q.astype(np.float32).reshape(-1,1)
val_age = np.expand_dims(np.load('val_age.npy'),1)
val_KPS = np.expand_dims(np.load('val_KPS.npy'),1)
val_gender = np.expand_dims(np.load('val_gender.npy'),1)

#load saved models
os.chdir('/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/data/data_splitted_balanced_20/outputs/models/')
model_FLAIR = load_model('flair_model_conv2.h5')
model_T2 = load_model('T2_model_conv2.h5')
model_T1 = load_model('T1_model_conv2.h5')
model_T1post = load_model('T1post_model_conv2.h5')

# obtain sigmoid probs for train set
train_sig_FLAIR = model_FLAIR.predict(train_FLAIR,batch_size=16)
train_sig_T2 = model_T2.predict(train_T2,batch_size=16)
train_sig_T1 = model_T1.predict(train_T1,batch_size=16)
train_sig_T1post = model_T1post.predict(train_T1post,batch_size=16)

# obtain sigmoid probs for val set
val_sig_FLAIR = model_FLAIR.predict(val_FLAIR,batch_size=16)
val_sig_T2 = model_T2.predict(val_T2,batch_size=16)
val_sig_T1 = model_T1.predict(val_T1,batch_size=16)
val_sig_T1post = model_T1post.predict(val_T1post,batch_size=16)

# hyperparameter for logistic: C value (i.e. regularization strength)
C_array = np.logspace(-2,2,5)

# Case1: Consider only the image sequences
train_imaging = np.hstack((train_sig_FLAIR,train_sig_T2,train_sig_T1,train_sig_T1post))
val_imaging = np.hstack((val_sig_FLAIR,val_sig_T2,val_sig_T1,val_sig_T1post))

best_penalty_type_imaging, best_c_imaging = hyper_parameter_search(train_feat= train_imaging, train_1p19q= train_1p19q, val_feat=val_imaging, val_1p19q=val_1p19q, C_array= C_array)

# Case2: Consider imaging + age
train_imaging_age = np.hstack((train_sig_FLAIR,train_sig_T2,train_sig_T1,train_sig_T1post, train_age))
val_imaging_age = np.hstack((val_sig_FLAIR,val_sig_T2,val_sig_T1,val_sig_T1post, val_age))

best_penalty_type_imaging_age, best_c_imaging_age = hyper_parameter_search(train_feat= train_imaging_age, train_1p19q= train_1p19q, val_feat=val_imaging_age, val_1p19q=val_1p19q, C_array= C_array)

# Case3: Consider imaging + age + KPS + gender
train_all = np.hstack((train_sig_FLAIR,train_sig_T2,train_sig_T1,train_sig_T1post, train_age, train_KPS, train_gender))
val_all = np.hstack((val_sig_FLAIR,val_sig_T2,val_sig_T1,val_sig_T1post, val_age, val_KPS, val_gender))

best_penalty_type_all, best_c_all = hyper_parameter_search(train_feat= train_all, train_1p19q= train_1p19q, val_feat=val_all, val_1p19q=val_1p19q, C_array= C_array)

#combine val and training features
train_val_imaging = np.vstack((train_imaging,val_imaging))
train_val_imaging_age = np.vstack((train_imaging_age,val_imaging_age))
train_val_all = np.vstack((train_imaging_all,val_imaging_all))

train_val_1p19q = np.append(train_1p19q, val_1p19q)

# train with best reg strength and penalty term
logreg_imaging = LogisticRegression(C=best_c_imaging, penalty = best_penalty_type_imaging, tol=0.01)
logreg_imaging_age = LogisticRegression(C=best_c_imaging_age, penalty = best_penalty_type_imaging_age, tol=0.01)
logreg_all = LogisticRegression(C=best_c_all, penalty = best_penalty_type_all, tol=0.01)

logreg_imaging.fit(train_val_imaging, train_val_1p19q)
logreg_imaging_age.fit(train_val_imaging_age, train_val_1p19q)
logreg_all.fit(train_val_all, train_val_1p19q)

# save the trained logistic regression classifiers
np.save('logreg_imaging.npy', logreg_imaging.__dict__)
np.save('logreg_imaging_age.npy', logreg_imaging_age.__dict__)
np.save('logreg_all.npy', logreg_all.__dict__)

# load test data
os.chdir('/rsrch1/bcb/Imaging-Genomics_SIBL/DONNIE_KIM/Brachy_deep_learning/IDH_Prediction/data/data_splitted_balanced_20/test')
test_FLAIR = np.load('test_FLAIR.npy')
test_T2 = np.load('test_T2.npy')
test_T1 = np.load('test_T1.npy')
test_T1post = np.load('test_T1post.npy')

test_1p19q = np.load('test_1p19q.npy')
test_age = np.expand_dims(np.load('test_age.npy'),1)
test_KPS = np.expand_dims(np.load('test_KPS.npy'),1)
test_gender = np.expand_dims(np.load('test_gender.npy'),1)

# obtain sigmoid probs for test set
test_sig_FLAIR = model_FLAIR.predict(test_FLAIR,batch_size=16)
test_sig_T2 = model_T2.predict(test_T2,batch_size=16)
test_sig_T1 = model_T1.predict(test_T1,batch_size=16)
test_sig_T1post = model_T1post.predict(test_T1post,batch_size=16)

test_imaging = np.hstack((test_sig_FLAIR,test_sig_T2, test_sig_T1, test_sig_T1post))
test_imaging_age = np.hstack((test_sig_FLAIR,test_sig_T2, test_sig_T1, test_sig_T1post, test_age))
test_all = np.hstack((test_sig_FLAIR,test_sig_T2, test_sig_T1, test_sig_T1post, test_age, test_KPS, test_gender))

Z_imaging = logreg_imaging.predict_proba(test_imaging)
Z_imaging_age = logreg_imaging_age.predict_proba(test_imaging_age)
Z_all = logreg_all.predict_proba(test_all)

imaging_acc = get_accuracy(test_1p19q,np.round(Z_imaging[:,1]))
imaging_sen = get_sensitivity(test_1p19q,np.round(Z_imaging[:,1]))
imaging_spe = get_specificity(test_1p19q,np.round(Z_imaging[:,1]))
get_auc(test_1p19q,Z_imaging[:,1])

imaging_age_acc = get_accuracy(test_1p19q,np.round(Z_imaging_age[:,1]))
imaging_age_sen = get_sensitivity(test_1p19q,np.round(Z_imaging_age[:,1]))
imaging_age_spe = get_specificity(test_1p19q,np.round(Z_imaging_age[:,1]))
get_auc(test_1p19q,Z_imaging_age[:,1])

all_acc = get_accuracy(test_1p19q,np.round(Z_all[:,1]))
all_sen = get_sensitivity(test_1p19q,np.round(Z_all[:,1]))
all_spe = get_specificity(test_1p19q,np.round(Z_all[:,1]))
get_auc(test_1p19q,Z_all[:,1])