import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append('/home/fendlnm1/Fendley/adversarial/ISC_AML_2018/src')
from evaluate_submissions import enforce_ell_infty_constraint


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
from cleverhans.attacks import FastGradientMethod

def main(args):

    ### Params of the run are here
    data_dir = args[1]
    output_dir = args[2]
    eps = args[3:]
    num_adv = 1000
    
    model = Sequential()
    model = load_model('cnn_image_only.model')
    model.compile(loss='categorical_crossentropy', optimizer='SGD',metrics=['accuracy'])

    # img_paths = prep_filelist(data_dir)
    # x_input, y_input, names = prep_adv_set(model,img_paths,num_adv=num_adv)
    # adv_fgsm(data_dir, output_dir, model, names, x_input, y_input=y_input,eps=eps)

    x_input, names = load_images(data_dir)
    adv_fgsm(data_dir, output_dir, model, names, x_input, eps=eps)
    

""" Given a category name returns the category ID
[ Params ]:
    name: the name of the category
[ Returns ]:
    categoryID: the unique ID of the correct decision
"""
def category_map(name):
    category_names = ['false_detection', 'airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'aquaculture', 'archaeological_site', 'barn', 'border_checkpoint', 'burial_site', 'car_dealership', 'construction_site', 'crop_field', 'dam', 'debris_or_rubble', 'educational_institution', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'flooded_road', 'fountain', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'interchange', 'lake_or_pond', 'lighthouse', 'military_facility', 'multi-unit_residential', 'nuclear_powerplant', 'office_building', 'oil_or_gas_facility', 'park', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'port', 'prison', 'race_track', 'railway_bridge', 'recreational_facility', 'impoverished_settlement', 'road_bridge', 'runway', 'shipyard', 'shopping_mall', 'single-unit_residential', 'smokestack', 'solar_farm', 'space_facility', 'stadium', 'storage_tank','surface_mine', 'swimming_pool', 'toll_booth', 'tower', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
    return category_names.index(name)


def load_images(data_dir):
    """
    Loads images from a challenge directory
    """
    images=os.listdir(data_dir)
    img_list = []
    filenames = []
    for img in images:
        if img.endswith('.png'):
            img_path = os.path.join(data_dir, img)
            img_pil = Image.open(img_path)
            x_input = np.asarray(img_pil).astype(np.float32)
            x_test = imagenet_preprocessing(x_input)
            img_list.append(x_test)
            filenames.append(img)

    return np.asarray(img_list), filenames



def adv_fgsm(data_dir, save_folder, model,filenames, x_input, y_input=None,eps=[0.01]):
    """ Attacks the fmow baseline model with an FGSM attack from cleverhans

    data_dir: directory with the image files
    eps: a list of perturbation constraints
    """
  
    sess = tf.Session()
    K.set_session(sess)
    K.set_learning_phase(0)
   

    sess.run(tf.global_variables_initializer())


    model = Sequential()
    model = load_model('cnn_image_only.model')
    model.compile(loss='categorical_crossentropy', optimizer='SGD',metrics=['accuracy'])

    #model.compile(loss='categorical_crossentropy', optimizer='SGD',metrics=['accuracy'])

    labels_all = []
    x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(None, 63))

    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap, sess=sess)

    # Generate perturbations at various epsilon
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        
    for ep in eps:
        print("[Info] Attacking epsilon " + ep)
        ep = int(ep)
        ep_float = float(ep)/255.0 
        fgsm_params = {'eps':ep_float}
        
        adv_x = fgsm.generate(x, **fgsm_params)
        eval_par = {'batch_size': 32}
        counter = 0
        hits = 0
        save_folder_eps = os.path.join(save_folder, str(ep))
        if not os.path.exists(save_folder_eps):
            os.mkdir(save_folder_eps)
        for i in range(len(x_input)):
            img_clean = np.expand_dims(x_input[i], axis=0)    
            img = adv_x.eval(session=sess, feed_dict={x:img_clean})
            
            if y_input is not None:
                cat = y_input[i]
                cat_input = np.expand_dims(cat, axis=0)

            img_out = prepare_image_output(img)

            ### Debugging the image clipping
            if ep != 0:
                img_clean_out = prepare_image_output(img_clean)
                img_out_crop = enforce_ell_infty_constraint(img_out, img_clean_out, ep)

                if not np.array_equal(img_out_crop, img_out):
                    print("Clipping was required")
                    img_out = img_out_crop

            img_PIL = Image.fromarray(img_out, 'RGB')
            img_PIL.save(os.path.join(save_folder_eps,filenames[i]))

            if ep == 0 and y_input is not None:
                labels_all.append([filenames[i],np.argmax(y_input[i])])
    
            if y_input is not None:
                pred = model.predict(img, batch_size=1)
                if np.argmax(pred) == np.argmax(cat_input):
                    hits += 1
                counter += 1

         
        if y_input is not None:
            print("[Info]: The accuracy on eps " + str(ep) + ': ' +str(float(hits)/counter))
            if ep == 0:
                print("Saving out labels")
                labels_np = np.asarray(labels_all)
                np.savetxt(os.path.join(save_folder,'labels.csv'), labels_np, fmt='%s', delimiter=',')
      


        

def prep_filelist(data_dir):
    """ Returns a list of all the filepaths of the png files

    data_dir: directory containing the .png files
    """
    file_paths = []
    # Create a cache to save time on rerun
    cache_path = 'cache/'
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)

    # Load all the image file paths into file_paths list
    if os.path.exists(os.path.join(cache_path,'all_filepaths.csv')):
        file_paths = np.genfromtxt(os.path.join(cache_path,'all_filepaths.csv'), dtype='str', delimiter=',')
    else:
        categories = os.listdir(data_dir)
        for cat in categories:
            cat_dir = os.path.join(data_dir,cat)
            images = os.listdir(cat_dir)
            for img_file in images:
                if img_file.endswith('.png'):
                    file_paths.append(os.path.join(cat_dir,img_file))
        # Save the list to the cache    
        print(" [ INFO ] : saving filepaths to cache")      
        np.savetxt(os.path.join(cache_path,'all_filepaths.csv'), file_paths, fmt='%s',delimiter=',')
    return file_paths


def imagenet_preprocessing(image):
    """ Function to perform imagenet preprocessing on image for densenet. It removes the imagenet mean and divides by 255,
        moving the image to [-1, 1]

    image: the image to move from [0,255] to [-1,1]
    """
    img = image.copy()
    mean = [103.939, 116.779, 123.68]

    img = img[..., ::-1]

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
    
    return np.squeeze(img.astype(np.uint8))


def prep_adv_set(model, filepaths, num_adv=1000, batch_size=256):
    """ Loads and returns images with their predictions that are correctly predicted by the provided model

    model: a keras model to evaluate images on
    filepaths: a list of filepaths to images
    num_adv: number of adversarial images to prepare
    """
    if num_adv == 'max':
        num_adv = len(filepaths)
    random.shuffle(filepaths)
    i = 0
    x_test_final = np.zeros((0,224,224,3))
    y_test_final = np.zeros((0,63))
    names_final = []
    
    print("[INFO]:  We are loading correctly detected images")
    while x_test_final.shape[0] < num_adv:
        if (len(filepaths) - i) < batch_size:
            batch_size = int(len(filepaths) - i)
            print(batch_size)
        x_batch = np.zeros((batch_size,224,224,3))
        y_batch = np.zeros((batch_size,63))
        names_batch = []
        for j in range(batch_size):
            img_path = filepaths[i]
            i += 1
            img_pil = Image.open(img_path)
            x_test = np.expand_dims(np.asarray(img_pil).astype(np.float32),axis =0)
            category = img_path.split('/')[-2]
            x_test = imagenet_preprocessing(x_test)
            y_test = to_categorical(category_map(category), 63)
            x_batch[j] = x_test
            y_batch[j] = y_test
            names_batch.append(os.path.basename(img_path))
        pred_batch = model.predict(x_batch, batch_size=batch_size)
        for idx in range(batch_size):
            if np.argmax(y_batch[idx]) == np.argmax(pred_batch[idx]):
                x_test_final = np.concatenate((x_test_final, np.expand_dims(x_batch[idx],axis=0)), axis = 0)
                y_test_final = np.concatenate((y_test_final, np.expand_dims(y_batch[idx],axis=0)), axis = 0)
                names_final.append(names_batch[idx])
        print ("[STATUS]:  On image " + str(i) + ', ' + str(x_test_final.shape[0]) + " good images loaded")
    x_test_final = x_test_final[:num_adv]
    y_test_final = y_test_final[:num_adv]
    names_final = names_final[:num_adv]
    print ("[Info ]: Finished loading good clean examples, found ", x_test_final.shape[0], "correct detections in ", i, "images")
    return x_test_final, y_test_final, names_final



if __name__ =='__main__':
    main(sys.argv)
