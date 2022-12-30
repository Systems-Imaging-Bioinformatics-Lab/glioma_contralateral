import os

def file_dirs(mode):
    if mode not in ('texture','TDA','CNN','cmb'):
        raise ValueError('Mode provided was not one of: (\'texture\',\'TDA\',\'CNN\',\'cmb\')')
    
    origdir = os.getcwd()
    basedir = os.path.abspath(os.path.join(origdir,'..','..'))
    imagedir = os.path.join(basedir,'data','N4_corrected_image_data') # what used to be called basedir
    normdir = os.path.join(basedir,'data','normalized_data_n4')
    splitdir = os.path.join(basedir,'data','data_split_n4')
    
    out_dir = os.path.join(splitdir,'outputs')
    model_dir = os.path.join(out_dir,'%s_models' % (mode))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    #return origdir,basedir,imagedir,normdir,splitdir,train_dir,val_dir,test_dir,out_dir,model_dir
    return origdir,basedir,imagedir,normdir,splitdir,out_dir,model_dir