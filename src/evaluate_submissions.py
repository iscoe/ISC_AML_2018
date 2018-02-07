""" Evaluate AE attacks and defenses.

 This script runs all attack submissions vs all defense submission and
 computes scores/metrics for each.

 Notes:
   o This general framework is inspired by the NIPS/Google Brain Kaggle
     competition from 2017.  While our attack format is a bit different,
     we adopt the same approach for defense.

   o All images are assumed to be of the form:
               #_rows, #_cols, #_channels
     i.e. tensorflow ordering.

   o We anticipate running this framework on a linux machine 
     with tensorflow and Python 3 (via anaconda) if you are 
     interested in replicating this environment locally. 
     We attempt to provide comments where this assumption
     is being made.

"""


__author__ = "mjp, nf"
__date__ = 'Feb 2018'

import sys
import os
import glob
import tempfile
from zipfile import ZipFile
import pdb
from functools import partial
from PIL import Image


import numpy as np



N_CLASSES = 70  # TODO: fix this!
ESTIMATES_FILE = 'estimates.csv'



# XXX: may update to use python logging package later
def _info(message):
    print("[INFO]: %s" % message)

def _warning(message):
    print('[WARNING]: %s' % message)
 
def _error(message):
    print('[ERROR]: %s' % message)


#-------------------------------------------------------------------------------
# Functions for working with the file-based "API" between us and the 
# web front end.
#-------------------------------------------------------------------------------

def _image_files(dir_name):
    "Returns a list of all valid (for our purposes) images in the given directory."
    # we require PNG images (due to lossless compression)
    return glob.glob(os.path.join(dir_name, '*.png'))


def _all_team_names(submission_dir):
    """ Returns a list of all team names (same as directory names).
    """
    dirnames = [x for x in os.listdir(submission_dir) if not x.startswith('.')]
    return dirnames


def _get_submission(team_dir, submission_type='attack'):
    """ Retrieves the most recent submission (a filename) from a team's directory.
    """
    submissions = []
    for extension in ["*.zip"]:  # for now, only zip (no tar/tgz)
        pattern = submission_type + extension
        submissions.extend(glob.glob(os.path.join(team_dir, pattern)))

    if len(submissions) == 0:
        return None

    # LINUX-SPECIFIC !!
    # We use the last modified time as the "creation" time.
    # Alternately, we could parse the timestamp in the filename?
    sub_time = [os.stat(x).st_mtime for x in submissions]
    newest_idx = np.argmax(np.array(sub_time))
    return submissions[newest_idx]



#-------------------------------------------------------------------------------
# Code related to attack submissions
#-------------------------------------------------------------------------------

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



def prepare_ae(ae_directory, tgt_directory, ref_directory, f_constraint):
    """ Ensures input images satisfy a maximum perturbation constraint.

       ae_directory  : Directory containing adversarial images
       tgt_directory : Directory where verified images should be copied.
       ref_directory : Directory containing clean/original images
       f_constraint  : A function f(x,y) which ensures image x is close enough to image y.

    """
    if not os.path.exists(ae_directory):
        _error('AE directory not found!')
        return
  
    cnt = 0
    for cnt, filename in enumerate(_image_files(ref_directory)):
        path, img_name = os.path.split(filename)
        x_ref = np.array(Image.open(filename), dtype=np.uint8)

        # load the corresponding image from source dir
        x_ae = np.array(Image.open(os.path.join(ae_directory, img_name)))

        # create the image that will actually be evaluated
        x_eval = f_constraint(x_ae, x_ref).astype(np.uint8)

        out_file = os.path.join(tgt_directory, img_name)
        Image.fromarray(x_eval, mode='RGB').save(out_file)

    if cnt == 0:
        _error('no reference files were found!  Is the setup correct?')


 
def _are_images_equivalent_p(dir_a, dir_b):
    """ Checks to see if the .png images in two directories contain
        equivalent signal content.
    """
    for filename in _image_files(dir_a):
        _, fn = os.path.split(filename)
        img_a = Image.open(filename)
        img_b = Image.open(os.path.join(dir_b, fn))
        if np.any(np.array(img_a) != np.array(img_b)):
            return False
        else:
            return True


#-------------------------------------------------------------------------------
# Code related to defense submissions
#-------------------------------------------------------------------------------

def load_estimates(csv_file_name):
    """ Reads class estimates from a .csv file.
    """
    file_names = []
    estimates = []
    
    with open(csv_file_name, 'r') as f:
        for line in f.readlines():
            pieces = [x.strip() for x in line.split(",")]
            labels = [int(x) for x in pieces[1:]]
            file_names.append(pieces[0])
            estimates.append(np.array(labels, dtype=np.int32))

    return file_names, np.array(estimates)



def _random_guessing_defense(input_dir, output_dir):
    """ Randomly guesses image class labels.

    This is just for shaking out the API; obviously this is not a suitable defense.
    """
    out_file = os.path.join(output_dir, ESTIMATES_FILE)
    
    with open(out_file, 'w') as f:
        for filename in _image_files(input_dir):
            path, fn = os.path.split(filename)
            guess = np.arange(N_CLASSES)
            np.random.shuffle(guess)
            line = fn + "," + ",".join([str(x) for x in guess])
            f.write(line + "\n")


#-------------------------------------------------------------------------------
#  Evauation (attack vs defense)
#-------------------------------------------------------------------------------

def run_attack_vs_defense(attack_zip, defense_zip, ref_dir, epsilon_values):
    """ Runs a single attack against a single defense.

    Produces a class estimate for every image/epsilon_value pair.
    """
    # Create temporary files (will be deleted when objects go out of scope)
    raw_dir = tempfile.TemporaryDirectory()          # we unzip attacker's images here
    def_in_dir = tempfile.TemporaryDirectory()       # images ready for defense live here
    def_out_dir = tempfile.TemporaryDirectory()      # output from defense goes here

    result = {}

    for epsilon in epsilon_values:
        f_con = partial(enforce_ell_infty_constraint, epsilon=epsilon)

        # unzip and prepare the attack images
        with ZipFile(attack_zip, 'r') as zf:
            zf.extractall(path=raw_dir.name)
            input_dir = os.path.join(raw_dir.name, str(epsilon))
            prepare_ae(input_dir, def_in_dir.name, ref_dir, f_con)
            if not _are_images_equivalent_p(input_dir, def_in_dir.name):
                _warning('input images did not satisfy constraints!!  They have been clipped accordingly.')

        # run defense on these images
        # TODO: nvidia-docker run goes here!
        pdb.set_trace() # TEMP
        _random_guessing_defense(def_in_dir.name, def_out_dir.name)
        file_names, Y_hat = load_estimates(os.path.join(def_out_dir.name, ESTIMATES_FILE))

        # store result
        result[epsilon] = (file_names, Y_hat)

    return result


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    y_true = 'TODO'  # TODO: implement this.
    submission_dir = '../misc/simulated_environment/submissions'  # TODO: fix this
    truth_dir = '../misc/simulated_environment/truth'
    epsilon_values_to_run = [10,]


    #--------------------------------------------------
    # Run all attacks vs all defenses
    #--------------------------------------------------
    all_results = {}  

    for attacker in _all_team_names(submission_dir):
        all_results[attacker] = {}

        attack_zip = _get_submission(os.path.join(submission_dir, attacker), 'attack')
        if attack_zip is None:
            continue # no attack from this team
        else:
            _info('processing attack file %s' % attack_zip)

        for defender in _all_team_names(submission_dir):
            # we do not play the same team's attack vs defense
            if defender == attacker:
                continue

            defense_zip = _get_submission(os.path.join(submission_dir, defender), 'defense')
            if defense_zip is None:
                continue # no defense from this team
            else:
                _info('processing defense file %s' % defense_zip)

            # try out an interaction between attack and defense

            result = run_attack_vs_defense(attack_zip, defense_zip, truth_dir, epsilon_values_to_run)
            all_results[attacker][defender] = result

    print(all_results) # TEMP
    compute_metrics(all_results, y_true)
