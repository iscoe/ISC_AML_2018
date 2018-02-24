"""  Example classifier/defense for FMoW dataset.

     Feel free to use this as a template for your own defenses.
     (this is not required, however; use any codes you like as
      long as we can run them in a public Docker)
"""

__author__ = "nf,mjp"
__date__ = 'feb 2018'


import sys
import random
import csv
import os
import glob
import pdb

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#import json

sys.path.append('./data_ml_functions/DenseNet')
from keras.models import Model, load_model, Sequential
import numpy as np
import tensorflow as tf
from keras.applications import imagenet_utils
from keras.utils.np_utils import to_categorical
from PIL import Image

from sklearn.metrics import accuracy_score, confusion_matrix



def main(args):
    """ Basic code to run the fMoW baseline classifier on a folder of images.
    """
    ### Import and create the basic fMoW baseline model
    #
    # one might also consider exploring different models...or ensembles of models...
    model = Sequential()
    model = load_model('cnn_image_only.model')
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    
    input_path = args[1]      # directory containing images to classify
    output_file = args[2]     # the csv file to write predictions
    
    # Load in the images from the specified folder 
    file_list = glob.glob(os.path.join(input_path, '*.png')) 
    file_list = [os.path.split(x)[-1] for x in file_list]  
    xTest = prep_test_images(file_list, input_path)


    # Use the classifier to generate predictions
    print(" [ INFO ]: Prediction on ", xTest.shape[0], " test images")
    if os.getenv('DEFENSE_METHOD') == 'simple-average':
        preds = simple_averaging_defense(model, xTest)
    else:
        preds = model.predict(xTest)


    # Generate and save csv of top 5 classes for each image
    preds_out = []
    for i in range(preds.shape[0]):
        preds_out.append([os.path.basename(file_list[i]),(np.argsort(preds[i])[::-1][:5])])
        
    csvfile = output_file
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for pred in preds_out:
            writer.writerow([pred[0],pred[1][0],pred[1][1],pred[1][2],pred[1][3],pred[1][4]])    

    #np.savetxt(output_file, preds_out, fmt='%s', delimiter=',')
    print(" [ INFO ]: wrote predictions to ", output_file)


    # (optional): if there was ground truth, see how we did
    labels_file = os.path.join(input_path, 'labels.csv')
    if os.path.exists(labels_file):
        y_hat = np.argmax(preds, axis=1)
        y_true = load_truth(labels_file, file_list)
        print('[info]: accuracy is %0.2f%%' % (100 * accuracy_score(y_true, y_hat)))



def simple_averaging_defense(model, x, epsilon=0.3, n_iters=5):
    """  Generate prediction by taking an average over a few pre-processed inputs.

         This is similar to the approach of 
             Cau & Gong "Mitigating Evasion Attacks..."

         There are many other "pre-processing style" defenses; e.g. see 
             Guo et al. "Countering Adversarial Images using Input Transformations."

         Of course, there are many other ways of implementing defenses that are not
         limited to manipulating the inputs...
    """
    def bernoulli_noise(x, epsilon):
        # note we are not clipping here - probably should...
        return x + epsilon * (np.random.randn(*x.shape) < 0.5)
    
    print('[info]: averaging over %d noisy estimates' % n_iters)

    preds = model.predict(bernoulli_noise(x, epsilon))
    for ii in range(n_iters-1):
        preds += model.predict(bernoulli_noise(x, epsilon))

    return preds / (1.*n_iters)



def prep_test_images(file_paths, direc):
    """ Function to prepare the images from the extracted bounding boxes
    [ Params ]:
        file_paths: list of all the file paths of the images
    [ Returns ]:
        xTest: an np array of the imagenet_mean subtracted data between [0,1]
    """ 
    num_images = len(file_paths)

    xTest = np.zeros((num_images,224,224,3))
    for i in range(num_images):
        img_path = file_paths[i]
        img_PIL = Image.open(os.path.join(direc,img_path))
        img = np.asarray(img_PIL)
        xTest[i] = img
    xTest = imagenet_utils.preprocess_input(xTest)
    xTest /= 255.
    return xTest




def category_map(name):
    """ Given a category name returns the category ID
    [ Params ]:
        name: the name of the category
    [ Returns ]:
        categoryID: the unique ID of the correct decision
    """
    category_names = ['false_detection', 'airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'aquaculture', 'archaeological_site', 'barn', 'border_checkpoint', 'burial_site', 'car_dealership', 'construction_site', 'crop_field', 'dam', 'debris_or_rubble', 'educational_institution', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'flooded_road', 'fountain', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'interchange', 'lake_or_pond', 'lighthouse', 'military_facility', 'multi-unit_residential', 'nuclear_powerplant', 'office_building', 'oil_or_gas_facility', 'park', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'port', 'prison', 'race_track', 'railway_bridge', 'recreational_facility', 'impoverished_settlement', 'road_bridge', 'runway', 'shipyard', 'shopping_mall', 'single-unit_residential', 'smokestack', 'solar_farm', 'space_facility', 'stadium', 'storage_tank','surface_mine', 'swimming_pool', 'toll_booth', 'tower', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
    return category_names.index(name)
  


def load_truth(labels_file, image_files):
    "Load true labels from .csv"
    if os.path.exists(labels_file):
        truth = {}
        with open(labels_file, 'r') as f:
            for line in f.readlines():
                filename, label = [x.strip() for x in line.split(",")]
                truth[filename] = int(label)
        ground_truth = [truth[f] for f in image_files]
    else:
        ground_truth = []
    return ground_truth



if __name__ == '__main__':
     main(sys.argv)
