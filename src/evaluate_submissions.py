""" Evaluate AE attacks and defenses.

 This script runs all attack submissions vs all defense submission and
 computes scores/metrics for each.

 Notes:
   o This general framework is inspired by the NIPS/Google Brain Kaggle
     competition from 2017.  While our attack format is different,
     we adopt the same approach for defense.

   o All images are assumed to be of the form:
               #_rows, #_cols, #_channels
     i.e. tensorflow ordering.

   o We anticipate running this framework on a linux machine 
     with tensorflow+keras and Anaconda Python 3 if one is
     interested in replicating this environment locally. 

"""


__author__ = "mjp, nf"
__date__ = 'Feb 2018'

import sys
import time
import datetime
import os
import glob
import shutil
import tempfile
from zipfile import ZipFile
import pwd
import grp
from functools import partial
import json
import subprocess
from stat import S_IRUSR,S_IWUSR,S_IRGRP,S_IWGRP,S_IROTH,S_IWOTH
from PIL import Image

import numpy as np
import pandas as pd

import pdb


ESTIMATES_FILE = 'labels.csv'            # name of file containing ground truth within test images directory
COMPETITION_UNTARGETED = "untargeted"    # tag identifying this particular competition




#-------------------------------------------------------------------------------
# misc. helper functions
#-------------------------------------------------------------------------------

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
    dirnames.sort()
    return dirnames


def _get_submission(team_dir, submission_type='attack'):
    """ Retrieves the most recent submission (a filename) from a team's directory.
    """
    submissions = []
    for extension in ["*.zip"]:  # for now, only zip (no tar/tgz)
        sub_dir = os.path.join(team_dir, submission_type)
        submissions.extend(glob.glob(os.path.join(sub_dir, extension)))

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
        os.makedirs(ae_directory)

    for ref_file in _image_files(ref_directory):

        # Each truth file should have a corresponding ae file.
        # UPDATE: If attacker neglected to provide this image, use reference file as fallback.
        path, img_name = os.path.split(ref_file)
        ae_file = os.path.join(ae_directory, img_name)

        if not os.path.exists(ae_file):
            _warning('AE file "%s" not found!  Using ref as backup.' % ae_file)
            shutil.copyfile(ref_file, ae_file)

        # load images and enforce perturbation constraint(s)
        x_ae = np.array(Image.open(ae_file), dtype=np.uint8)
        x_ref = np.array(Image.open(ref_file), dtype=np.uint8)
        x_eval = f_constraint(x_ae, x_ref).astype(np.uint8)  # enforce constraint

        # save admissible image
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



def run_defense(defense_dir, offense_dir, output_dir):
    """ Executes a defense on a collection of images.
    """

    #Load metadata from their submission
    metadata = json.load(open(os.path.join(defense_dir,'metadata.json')))
    outputname = '/output/predictions.csv'
    

    cmd = ['sudo', 'chown','1005:1005',defense_dir]
    subprocess.call(cmd)
    cmd = ['sudo', 'chmod','775',defense_dir]
    subprocess.call(cmd)
    cmd = ['sudo', 'chown','1005:1005',offense_dir]
    
    subprocess.call(cmd)
    cmd = ['sudo', 'chmod','775',offense_dir]
    subprocess.call(cmd)
    cmd = ['sudo', 'chown','33:33',output_dir]
    subprocess.call(cmd)
    cmd = ['sudo', 'chmod','775',output_dir]
    subprocess.call(cmd)


    for file in os.listdir(defense_dir):
        if file.endswith('.sh'):
            cmd = ['sudo', 'chmod','+x',os.path.join(defense_dir, file)]
            subprocess.call(cmd)
        
        # cmd = ['sudo', 'chown','1005:1005',os.path.join(defense_dir, file)]
        # subprocess.call(cmd)
        # cmd = ['sudo', 'chmod','775',os.path.join(defense_dir, file)]
        # subprocess.call(cmd)


    # cmd = ['sudo', 'chown','www-data:www-data',os.path.join(defense_dir, file)]
    # subprocess.call(cmd)
    
    #create nvidia docker command to run
    cmd = ['sudo', 'nvidia-docker', 'run',
           '-v', '{0}:/input_images'.format(offense_dir),
           '-v', '{0}:/output'.format(output_dir),
           '-v', '{0}:/code'.format(defense_dir),
           '-w', '/code',
           '--user', 'www-data', metadata['container_gpu'], metadata['entry_point'],
           '/input_images', outputname]
    
    subprocess.call(cmd)


#-------------------------------------------------------------------------------
#  Evauation (i.e. run attack vs defense)
#-------------------------------------------------------------------------------

def run_one_attack_vs_one_defense(attacker_id, attack_zip, defender_id, defense_zip, ref_dir, epsilon_values):
    """ Runs a single attack against a single defense.

        Note this evaluates this attack/defense pair for all values of epsilon.
    """

    # ground truth
    test_files, y_true = load_estimates(os.path.join(ref_dir, ESTIMATES_FILE))
    n_test = len(test_files)
    assert(y_true.size == n_test)  # truth file should have only one label

    # Extract attack submission (contains images for all epsilon values) 
    raw_dir = tempfile.mkdtemp()          # we unzip attacker's images here
    with ZipFile(attack_zip, 'r') as zf:
        zf.extractall(path=raw_dir)

    # Extract defense submission (executable code)
    def_dir = tempfile.mkdtemp()
    with ZipFile(defense_zip, 'r') as zf:
        zf.extractall(path=def_dir)

    results = []
    
    for epsilon in epsilon_values:
        f_con = partial(enforce_ell_infty_constraint, epsilon=epsilon) # enforces constraint
        def_in_dir = tempfile.mkdtemp()       # images ready for defense live here
        def_out_dir = tempfile.mkdtemp()      # output from defense goes here

        #----------------------------------------
        # prepare the attack images for this value of epsilon
        #----------------------------------------
        input_dir = os.path.join(raw_dir, str(epsilon))
        prepare_ae(input_dir, def_in_dir, ref_dir, f_con)
        if not _are_images_equivalent_p(input_dir, def_in_dir):
            _warning('input images did not satisfy constraints!!  They have been clipped accordingly.')
        attack_files = [os.path.basename(x) for x in _image_files(def_in_dir)]  # list of files created by attacker

        #----------------------------------------
        # run defense on these images
        #----------------------------------------
        run_defense(def_dir, def_in_dir, def_out_dir)
        defense_files, Y_hat = load_estimates(os.path.join(ref_dir, ESTIMATES_FILE))

        #----------------------------------------
        # evaluate performance
        #----------------------------------------
        score = np.zeros((n_test,))

        for ii in range(len(test_files)):
            fn, y_i = test_files[ii], y_true[ii]

            # top-1 accuracy; 1 denotes a success by the defense
            idx = defense_files.index(fn)
            y_hat_i = Y_hat[idx,0]
            score[ii] = 1 if (y_hat_i == y_i) else 0

        results.append((COMPETITION_UNTARGETED, attacker_id, defender_id, epsilon) + tuple(score))

    cols = ['competition', 'attacker-id', 'defender-id', 'epsilon'] + test_files
    return pd.DataFrame(results, columns=cols)



def run_attacks_vs_defenses(submission_dir, truth_dir, epsilon_values):
    """ Runs each attack vs each defense.
    """
    all_results = []

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
            result_this_pair = run_one_attack_vs_one_defense(attacker_id, attack_zip, defender_id, defense_zip, truth_dir, epsilon_values)
            all_results.append(result_this_pair)
            #except Exception as ex:
            #    _warning('%s vs %s failed! %s' % (attacker_id, defender_id, str(ex)))

    return pd.concat(all_results)



def compute_metrics(results, out_dir):
    """  Summarizes overall results and saves to .csv.
    """

    # Create output subdirectories (if needed)
    # TODO: update these as needed for actual directory structure
    out_dir_attack = os.path.join(out_dir, 'attack')
    if not os.path.isdir(out_dir_attack):
        os.makedirs(out_dir_attack)

    out_dir_defense = os.path.join(out_dir, 'defense')
    if not os.path.isdir(out_dir_defense):
        os.makedirs(out_dir_defense)

    n_images = results.shape[1] - 4

    #----------------------------------------
    # compute average performance (per test image)
    #----------------------------------------
    results_def = results.reset_index().groupby("defender-id").mean()
    results_def = results_def.drop(['epsilon', 'index'], axis=1)

    results_att = -results.reset_index().groupby("attacker-id").mean()
    results_att = results_att.drop(['epsilon', 'index'], axis=1)

    results_def.to_csv(os.path.join(out_dir_defense, 'details.csv'), header=True)
    results_att.to_csv(os.path.join(out_dir_attack, 'details.csv'), header=True)

    #----------------------------------------
    # compute net performance (avg. across all test images)
    #----------------------------------------
    results_def_agg = results_def.mean(axis=1)
    results_att_agg = results_att.mean(axis=1)

    results_def_agg.to_csv(os.path.join(out_dir_defense, "scores.csv"), header=True)
    results_att_agg.to_csv(os.path.join(out_dir_attack, "scores.csv"), header=True)


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    #----------------------------------------
    # command line parameters
    #----------------------------------------
    submission_dir = sys.argv[1]
    truth_dir = sys.argv[2]
    output_dir = sys.argv[3]
    epsilon_values_to_run = [float(x) for x in sys.argv[4:]]

    if not os.path.isdir(submission_dir):
        raise RuntimeError('Invalid submission directory: "%s"' % submission_dir)
    
    if not os.path.isdir(truth_dir):
        raise RuntimeError('Invalid truth directory: "%s"' % truth_dir)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # we'll store all results from this run in a separate, timestamped directory
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    output_dir_ts = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir_ts)

    #----------------------------------------
    # run attacks vs defenses
    #----------------------------------------
    tic = time.time()
    results = run_attacks_vs_defenses(submission_dir, truth_dir, epsilon_values_to_run)
    runtime = time.time() - tic
    _info('evaluation ran in %0.2f minutes' % (runtime/60.))

    fn = os.path.join(output_dir_ts, 'results.pkl')
    results.to_pickle(fn)

    #----------------------------------------
    # generate feedback for performers
    #----------------------------------------
    compute_metrics(results, output_dir_ts)
