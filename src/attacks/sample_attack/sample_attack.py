"""  Example attacks against FMoW image-only dataset.

     Feel free to use this as a template for your own attacks.
     (this is not required however; you can use any codes you like)
"""

__author__ = 'nf,mjp'
__date__ = 'february 2018'


import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # change this if you want to use a GPU other than 0

import json
import numpy as np
from PIL import Image
import random
import sys
import pdb

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod # there are other methods...

from sklearn.metrics import accuracy_score, confusion_matrix



def main(args):
    ### Params of the run are here
    data_dir = args[1]       # directory containing images to attack
    output_dir = args[2]     # directory where AE should be placed
    attack_type = args[3]    # attack type to use
    eps = args[4:]           # the epsilon values to use

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Load all images to attack in to memory at once.
    # This presumes a relatively small set of images (as will be the case in the competition).
    x_input, names, y_input = load_images(data_dir)
    print('[info]: Attacking %d images using method "%s"' % (x_input.shape[0], attack_type))
    print('[info]: x min/max  %0.2f / %0.2f' % (np.min(x_input), np.max(x_input)))

    # Run the attack!
    adv_cleverhans(data_dir, output_dir, names, x_input, attack_type, y_input=y_input, eps=eps)




def load_images(data_dir):
    """
    Loads images from a challenge directory
    """
    # load images
    images=os.listdir(data_dir)
    img_list = []
    filenames = []
    for img in images:
        if img.endswith('.png'):
            img_path = os.path.join(data_dir, img)
            img_pil = Image.open(img_path)
            x_input = np.asarray(img_pil,dtype=np.uint8)
            img_list.append(x_input) 
            filenames.append(img)

    # if there are also class labels, load those as well
    labels_file = os.path.join(data_dir, 'labels.csv')
    if os.path.exists(labels_file):
        print('[info]: also loading ground truth!!')
        truth = {}
        with open(labels_file, 'r') as f:
            for line in f.readlines():
                filename, label = [x.strip() for x in line.split(",")]
                truth[filename] = int(label)
        ground_truth = [truth[f] for f in filenames]
    else:
        ground_truth = []

    return np.asarray(img_list), filenames, np.array(ground_truth, dtype=np.int32)



def adv_cleverhans(data_dir, save_folder, filenames, x_input, attack_type, y_input, eps):
    """ Attacks the fmow baseline model with an FGSM attack from cleverhans

    data_dir: directory with the image files
    eps: a list of perturbation constraints
    """
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    sess = tf.Session()
    K.set_session(sess)
    K.set_learning_phase(0)
   
    sess.run(tf.global_variables_initializer())

    #
    # Create the model we will attack.
    # Here, this is the FMoW classifier based on DenseNets.
    # Of course, one could attack something else (adversarially trained network, an ensemble, etc.)
    #
    model = Sequential()
    model = load_model('cnn_image_only.model')
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

    labels_all = []
    x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(None, 63))

    #--------------------------------------------------
    # Use CleverHans (CH) to generate the attack.
    # Note that the CH API wants a "Model" object.
    # For Keras models, there is a convenience "wrapper" to help with this.
    #--------------------------------------------------
    wrap = KerasModelWrapper(model)
    if attack_type.lower() in ['fgm', 'random']:
        attack = FastGradientMethod(wrap, sess=sess) 
    elif attack_type.lower() == 'ifgm':
        attack = BasicIterativeMethod(wrap, sess=sess)
    else:
        # one could try other methods here...
        raise ValueError('unsupported attack type', attack_type)


    # Generate perturbations at various epsilon
    for ep in eps:
        print("[info]: Attacking epsilon " + ep)
        ep = int(ep)
        ep_float = float(ep)/255.0        # epsilon value in the network's input space

        save_folder_eps = os.path.join(save_folder, str(ep))
        if not os.path.exists(save_folder_eps):
            os.mkdir(save_folder_eps)

        #--------------------------------------------------
        # Create Tensorflow variable for the adversarial example and this value of epsilon.
        # Note that different attacks may have different parameters to experiment with...
        #--------------------------------------------------
        attack_params = {'eps':ep_float}
        if attack_type.lower() == 'ifgm':
            attack_params['nb_iter'] = 5   # number of iterations
        adv_x = attack.generate(x, **attack_params)

        #--------------------------------------------------
        # process images one at a time (note: it is possible to attack batches as well...)
        #--------------------------------------------------
        y_hat = -1 * np.ones((len(x_input),), dtype=np.int32)
        for i in range(len(x_input)):
            if attack_type.lower() == 'random':
                # special case: a naive random perturbation attack
                img_out = random_perturbation(x_input[i], ep)
                img = imagenet_preprocessing(img_out)
                img = img[np.newaxis, ...]
            else:
                # using a CleverHans attack
                img_clean = np.expand_dims(x_input[i], axis=0)            # add (trivial) mini-batch dimension
                img_clean = imagenet_preprocessing(img_clean)             # preprocessing assumed by baseline classifier
                img = adv_x.eval(session=sess, feed_dict={x:img_clean})   # AE in the input space of the baseline classifier
                img_out = prepare_image_output(img)                       # AE in the original input space [0,255]

            # verify that epsilon constraint is satisfied
            delta = np.max(np.abs(x_input[i] - 1.0*img_out))
            assert(delta <= (ep + 1e-6))

            # Save resulting AE to file
            img_PIL = Image.fromarray(img_out, 'RGB')
            img_PIL.save(os.path.join(save_folder_eps,filenames[i]))

            # Performance of baseline classifier on resulting AE.
            # This is just for debugging purposes (not technically required).
            y_hat[i] = np.argmax(model.predict(img, batch_size=1))

        
        #--------------------------------------------------
        # (optional): report AE performance, if we know ground truth
        #--------------------------------------------------
        if len(y_input):
            print('[info]: baseline classifier accuracy on (%s,eps=%d) is %0.2f%%' % (attack_type, ep, 100 * accuracy_score(y_input, y_hat)))

        

def imagenet_preprocessing(image):
    """ Pre-processes images for use with the FMoW baseline.
   
    There are many "standard" ways of pre-processing images for use with CNNs.
    In this case, the FMoW folks elected to use a "caffe-style" preprocessing
    which, for original inputs in [0,255], subtracts a mean value and shuffles
    the color channels.

    In lieu of whitening the data (dividing by std) they developers elected
    to scale the translated data by 255.  We follow suit here; 
    note, however, that it is usually more convenient when working with
    AE to keep the values in some known space, like [0,1] or [-1,1] so that
    clipping is more straightforward.
    """
    img = image.copy().astype(np.float32)
    mean = [103.939, 116.779, 123.68]

    img = img[..., ::-1]  # shuffle RGB

    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]

    img /= 255.0

    return img


def prepare_image_output(image):
    """ Undoes the imagenet preprocessing in order to output a clean image

    image: image to be moved to [0, 255] that has been imagenet preprocessed
    """
    img = image.copy()
    mean = [103.939, 116.779, 123.68]

    img *= 255.0
    
    img[..., 0] += mean[0]
    img[..., 1] += mean[1]
    img[..., 2] += mean[2]
    
    img = img[..., ::-1]
    img = np.clip(img, 0, 255)
    return np.squeeze(img).astype(np.uint8)



def random_perturbation(x_input, eps):
    """ Generates a input perturbation uniformly at random.
        This is a poor method for generating AE; only exists for pedagogical purposes.

        x_input : an image in the input space [0,255]^d
        epsilon : the maximum \ell_\infty perturbation.
    """
    x_adv = x_input + eps*np.sign(np.random.rand(*x_input.shape) - 0.5)
    # clip to feasible region
    x_adv[x_adv < 0] = 0
    x_adv[x_adv > 255] = 255
    return x_adv



if __name__ =='__main__':
    main(sys.argv)
