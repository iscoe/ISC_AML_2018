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
import pandas as pd



N_CLASSES = 70  # TODO: fix this!
ESTIMATES_FILE = 'labels.csv'
ATTACK_TAG = "attack"
DEFENSE_TAG = "defense"


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
        _info('No submission type "%s" in directory "%s"' % (submission_type, team_dir))
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
        _warning('AE directory "%s" not found - did attack not address this epsilon?' % ae_directory)
        return
  
    for filename in _image_files(ae_directory):
        path, img_name = os.path.split(filename)
        x_ae = np.array(Image.open(filename), dtype=np.uint8)

        # load the corresponding image from reference directory
        ref_file = os.path.join(ref_directory, img_name)
        if not os.path.exists(ref_file):
            _warning('no such reference file "%s"; ignoring this image' % ref_file)
            continue
        x_ref = np.array(Image.open(ref_file))

        # create the image that will actually be evaluated
        x_eval = f_constraint(x_ae, x_ref).astype(np.uint8)

        out_file = os.path.join(tgt_directory, img_name)
        Image.fromarray(x_eval, mode='RGB').save(out_file)


 
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
    return True


#-------------------------------------------------------------------------------
# Code related to defense submissions
#-------------------------------------------------------------------------------

def load_estimates(csv_file_name):
    """ Reads class estimates (or truth) from a .csv file.

        File format is assumed to be:

        image_filename_1.png, y_1, [y_2, y_3, ...]
        image_filename_2.png, y_1, [y_2, y_3, ...]
        ...
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

def run_one_attack_vs_one_defense(attacker_id, attack_zip, defender_id, defense_zip, ref_dir, epsilon_values):
    """ Runs a single attack against a single defense.

    Produces a class estimate for every image/epsilon_value pair.
    """

    # ground truth
    test_files, y_true = load_estimates(os.path.join(ref_dir, ESTIMATES_FILE))
    n_test = len(test_files)
    assert(y_true.size == n_test)  # truth file should have only one label

    # Extract attack submission (only need to do this once - contains all epsilon) 
    raw_dir = tempfile.TemporaryDirectory()          # we unzip attacker's images here
    with ZipFile(attack_zip, 'r') as zf:
        zf.extractall(path=raw_dir.name)

    results_attacker = []
    results_defender = []
    
    for epsilon in epsilon_values:
        f_con = partial(enforce_ell_infty_constraint, epsilon=epsilon) # enforces constraint
        def_in_dir = tempfile.TemporaryDirectory()       # images ready for defense live here
        def_out_dir = tempfile.TemporaryDirectory()      # output from defense goes here

        #----------------------------------------
        # prepare the attack images for this value of epsilon
        #----------------------------------------
        input_dir = os.path.join(raw_dir.name, str(epsilon))
        prepare_ae(input_dir, def_in_dir.name, ref_dir, f_con)
        if not _are_images_equivalent_p(input_dir, def_in_dir.name):
            pdb.set_trace() # TEMP
            _warning('input images did not satisfy constraints!!  They have been clipped accordingly.')
        attack_files = [os.path.basename(x) for x in _image_files(def_in_dir.name)]  # list of files created by attacker

        #----------------------------------------
        # run defense on these images
        #----------------------------------------
        # TODO: nvidia-docker run goes here!
        _random_guessing_defense(def_in_dir.name, def_out_dir.name)
        defense_files, Y_hat = load_estimates(os.path.join(def_out_dir.name, ESTIMATES_FILE))

        #----------------------------------------
        # evaluate performance
        #----------------------------------------
        attacker_score = np.zeros((n_test,))
        defender_score = np.zeros((n_test,))

        for ii in range(len(test_files)):
            fn, y_i= test_files[ii], y_true[ii]

            # Special cases: attacker/defender does not produce/evaluate an example
            #                Do not penalize the other party in this case (use NaN)
            if fn not in attack_files:
                attacker_score[ii] = 0
                defender_score[ii] = np.nan
                continue
            elif fn not in defense_files:
                attacker_score[ii] = np.nan
                defender_score[ii] = 0
                continue

            # top-1 accuracy
            idx = defense_files.index(fn)
            y_hat_i = Y_hat[idx,0]
            if y_hat_i == y_i:
                attacker_score[ii] = 0
                defender_score[ii] = 1
            else:
                attacker_score[ii] = 1
                defender_score[ii] = 0

        results_attacker.append((ATTACK_TAG, attacker_id, defender_id, epsilon) + tuple(attacker_score))
        results_defender.append((DEFENSE_TAG, attacker_id, defender_id, epsilon) + tuple(defender_score))

    return results_attacker, results_defender



def run_attacks_vs_defenses(submission_dir, truth_dir, epsilon_values):
    """ Runs each attack vs each defense.
    """
    all_attack = []
    all_defense = []

    for attacker_id in _all_team_names(submission_dir):
        #----------------------------------------
        # Get the attack submission
        #----------------------------------------
        _info('processing attacker: "%s"' % attacker_id)
        attack_zip = _get_submission(os.path.join(submission_dir, attacker_id), 'attack')
        if attack_zip is None:
            continue # no attack from this team

        for defender_id in _all_team_names(submission_dir):
            #----------------------------------------
            # get the defense submission
            #----------------------------------------
            if defender_id == attacker_id:  # do not play same team's attack vs defense
                continue

            _info('processing : "%s" vs "%s"' % (attacker_id, defender_id))
            defense_zip = _get_submission(os.path.join(submission_dir, defender_id), 'defense')
            if defense_zip is None:
                continue # no defense submission from this team

            #----------------------------------------
            # run attack vs defense and store result
            #----------------------------------------
            #try:
            attack_result, defense_result = run_one_attack_vs_one_defense(attacker_id, attack_zip, defender_id, defense_zip, truth_dir, epsilon_values)
            all_attack.append(attack_result)
            all_defense.append(defense_result)
            #except Exception as ex:
            #    _warning('%s vs %s failed! %s' % (attacker_id, defender_id, str(ex)))

    return pd.DataFrame(all_attack), pd.DataFrame(all_defense)



def compute_metrics(all_attacks, all_defenses):
    # TODO: save out some .csv files
    pass


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    submission_dir = sys.argv[1]
    truth_dir = sys.argv[2]
    epsilon_values_to_run = [int(x) for x in sys.argv[3:]]

    if not os.path.isdir(submission_dir):
        raise RuntimeError('Invalid submission directory: "%s"' % submission_dir)
    
    if not os.path.isdir(truth_dir):
        raise RuntimeError('Invalid truth directory: "%s"' % truth_dir)

    attacks, defenses = run_attacks_vs_defenses(submission_dir, truth_dir, epsilon_values_to_run)
    
    compute_metrics(attacks, defenses)
