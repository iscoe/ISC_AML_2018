
""" Evaluate AE attacks and defenses.

 This script runs all attack submissions vs all defense submission and
 computes scores/metrics for each.

 All images are assumed to be of the form (#_rows, #_cols, #_channels)
 (i.e. tensorflow ordering).

 Note: We assume Python 3.
"""


__author__ = "mjp, nf"
__date__ = 'Feb 2018'

import sys, os, glob
import pdb
from functools import partial
from PIL import Image


import numpy as np




def enforce_ell_infty_constraint(x_ae, x_orig, epsilon, clip_min=0, clip_max=255):
    """ Returns a copy of x_ae that satisfies 

        || x_ae - x_orig ||_\infty \leq epsilon,  clip_min <= x_ae,  x_ae <= clip_max

    x_ae    : a numpy tensor corresponding to the adversarial/perturbed image
    x_orig  : a numpy tensor corresponding to the original image
    epsilon : the perturbation constraint (scalar)
    """
    delta = x_orig - x_ae
    delta = np.clip(delta, -epsilon, epsilon)
    return np.clip(x_orig + delta, clip_min, clip_max)

 

def prepare_ae(ae_directory, ref_directory, tgt_directory, f_constraint):
    """ Ensures input images satisfy a maximum perturbation constraint.

       ae_directory  : Directory containing adversarial images
       ref_directory : Directory containing clean/original images
       tgt_directory : Directory where verified images should be copied.
       f_constraint  : A function f(x,y) which ensures image x is close enough to image y.

    """
    for filename in glob.glob(os.path.join(ref_directory, '*png')):
        path, img_name = os.path.split(filename)
        x_ref = np.array(Image.open(filename), dtype=np.uint8)

        # load the corresponding image from source dir
        x_ae = np.array(Image.open(os.path.join(ae_directory, img_name)))

        # create the image that will actually be evaluated
        x_eval = f_constraint(x_ae, x_ref).astype(np.uint8)

        out_file = os.path.join(tgt_directory, img_name)
        Image.fromarray(x_eval, mode='RGB').save(out_file)


        
def _are_images_equivalent_p(dir_a, dir_b):
    """ Checks to see if the .png images in two directories contain
        equivalent signal content.
    """
    for filename in glob.glob(os.path.join(dir_a, '*.png')):
        _, fn = os.path.split(filename)
        img_a = Image.open(filename)
        img_b = Image.open(os.path.join(dir_b, fn))
        if np.any(np.array(img_a) != np.array(img_b)):
            return False
        else:
            return True



if __name__ == "__main__":
    #input_dir = sys.argv[1]  # TODO

    ae_dir = './foo'
    ref_dir = './test_images'
    tgt_dir = '/tmp/foo'

    f_ell_infty = partial(enforce_ell_infty_constraint, epsilon=10)

    prepare_ae(ae_dir, ref_dir, tgt_dir, f_ell_infty)
    assert(_are_images_equivalent_p(ae_dir, tgt_dir))  # no changes expected

    #try:
    #    prepare_ae('./foo', './test_images', '/tmp/foo', f_con)
    #except Exception as ex:
    #    print('[ERROR]: pair "%s vs %s" failed due to %s' % ('foo', 'bar', str(ex)))
