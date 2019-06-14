import numpy as np
def get_pixel_area(segm):

    return segm.shape[0] * segm.shape[1]



def extract_both_masks(eval_segm, gt_segm, cl, n_cl):

    eval_mask = extract_masks(eval_segm, cl, n_cl)

    gt_mask   = extract_masks(gt_segm, cl, n_cl)



    return eval_mask, gt_mask



def extract_classes(segm):

    cl = np.unique(segm)

    n_cl = len(cl)



    return cl, n_cl



def union_classes(eval_segm, gt_segm):

    eval_cl, _ = extract_classes(eval_segm)

    gt_cl, _   = extract_classes(gt_segm)



    cl = np.union1d(eval_cl, gt_cl)

    n_cl = len(cl)



    return cl, n_cl



def extract_masks(segm, cl, n_cl):

    h, w  = segm_size(segm)

    masks = np.zeros((n_cl, h, w))



    for i, c in enumerate(cl):

        masks[i, :, :] = segm == c



    return masks



def segm_size(segm):

    try:

        height = segm.shape[0]

        width  = segm.shape[1]

    except IndexError:

        raise



    return height, width



def check_size(eval_segm, gt_segm):

    h_e, w_e = segm_size(eval_segm)

    h_g, w_g = segm_size(gt_segm)



    if (h_e != h_g) or (w_e != w_g):

        raise EvalSegErr("DiffDim: Different dimensions of matrices!")



'''

Exceptions

'''

class EvalSegErr(Exception):

    def __init__(self, value):

        self.value = value



    def __str__(self):

        return repr(self.value)