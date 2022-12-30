import numpy as np
import scipy.ndimage as ndimage
from skimage import segmentation
from skimage.segmentation import mark_boundaries
from scipy.interpolate import pchip_interpolate
import pywt

def adjust_image_contour(img_slice,mask_slice):
    nVox = img_slice.size
    nSegs = np.int(nVox/64)
    rsIm = pchip_interpolate((0,1),(np.min(img_slice),np.max(img_slice)),img_slice)
    sp_im = segmentation.slic(rsIm,nSegs,sigma = 2)

    n_per_label = np.bincount(sp_im.ravel())
    adj_segs = (sp_im+1)*mask_slice
    n_ov_per_label = np.bincount(adj_segs.ravel(),minlength= sp_im.max()+2)
    n_ov_per_label = n_ov_per_label[1:]

    pOv_lbl = n_ov_per_label/n_per_label

    outMask = np.zeros(img_slice.shape)
    randVal = np.random.random(pOv_lbl.shape)
    maxOv = pOv_lbl.max()
    for i in range(len(n_per_label)):
        idxs = np.argwhere(sp_im==i)
        currP = pOv_lbl[i]
        if currP == maxOv:
            outMask[idxs[:,0],idxs[:,1]] = 1
        elif currP >= .9:
            outMask[idxs[:,0],idxs[:,1]] = 1
        elif currP >= .2 and currP < .9:
            if currP > randVal[i]:
                outMask[idxs[:,0],idxs[:,1]] = 1
    outMask = ndimage.binary_closing(outMask) * 1
    return outMask

def get_noise(img_slice):
    (cA, (cH, cV, cD)) = pywt.dwt2(img_slice, 'coif1')
    sigma_est = np.median(abs(cD))/(.6754)
    return sigma_est

def add_noise(img_slice,reps=1,sigma=None):
    if sigma == None:
        sigma_est = get_noise(img_slice)
    else:
        sigma_est = sigma
    for r in range(reps):
        noise_slice = np.random.normal(loc=0,scale=sigma_est,size=img_slice.shape)
        img_slice = img_slice + noise_slice
    return img_slice

def grow_shrink_image(mask_slice,tau_factor):
    # slightly different than the described process in the paper, which used iterative 6 neighbor erosion/dilation
    # might need to add in a random element in the ordering step
    dims = mask_slice.shape
    n_voxels = np.sum(mask_slice>0)
    n_target_vox = np.int(np.round(n_voxels*(1+tau_factor)))
    
    inv_mask_slice =(~mask_slice.astype(bool)).astype(int) # invert the slice
    pos_bwdist = ndimage.morphology.distance_transform_edt(inv_mask_slice)
    neg_bwdist = ndimage.morphology.distance_transform_edt(mask_slice)
    bwdist = pos_bwdist-neg_bwdist # positive distance = growth, negative distance = shrink
    ord_bw_idxs = np.argsort(bwdist.ravel()) # order the pixels based on their distance to a part of the mask
    ord_bw_subs = np.unravel_index(ord_bw_idxs, dims, order='C')
    ord_bw_im = np.reshape(ord_bw_idxs,dims)
    
    outIm = np.zeros(dims).ravel()
    outIm[ord_bw_idxs[:n_target_vox]] = 1
    outIm = np.reshape(outIm,dims)
    return outIm

def rotate_image(img_slice, angle, mask_slice):
    com = ndimage.measurements.center_of_mass(mask_slice)
    pivot = np.int_(np.round(com))
    padX = [img_slice.shape[1] - pivot[0], pivot[0]]
    padY = [img_slice.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img_slice, [padY, padX], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

def translate_image(img_slice,shift):
    dim = img_slice.shape
    xIV = range(dim[1])
    yIV = range(dim[0])
    (xm,ym) = np.meshgrid(xIV, yIV)
    shift = [-22.5, 12.5]
    tr_im = ndimage.map_coordinates(img_slice,[ym-shift[1], xm-shift[0]])
    return tr_im

