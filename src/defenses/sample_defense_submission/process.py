#This is the file to compile and create the model
import sys
import redis
import random
import csv
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json

sys.path.append('./data_ml_functions/DenseNet')
from keras.models import Model, load_model, Sequential
import numpy as np
import tensorflow as tf
from keras.applications import imagenet_utils
from keras.utils.np_utils import to_categorical
from PIL import Image

""" Basic code to run the fMoW baseline classifier on a folder of jpgs, and will write a csv out to the file specified
[ Params ]:
    args[1]: The path to the directory of images
    args[2]: the csv file to write the predictions
"""
def main(args):
    ### Import and create the basic fMoW baseline model
    model = Sequential()
    model = load_model('cnn_image_only.model')
    model.compile(loss='categorical_crossentropy', optimizer='SGD',metrics=['accuracy'])
    
    input_path= args[1]
    output_file = args[2]
    
    # Load in the images from the specified folder while preserving order
    file_list = os.listdir(input_path)
    xTest = prep_test_images(file_list,input_path)
        
    
    # Use the classifier to generate predictions
    print(" [ INFO ]: Prediction on ", xTest.shape[0], " test images")
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
    
""" Function to prepare the images from the extracted bounding boxes
[ Params ]:
    file_paths: list of all the file paths of the images
    num_images: number of images to prepare, default 1000
[ Returns ]:
    xTest: an np array of the imagenet_mean subtracted data between [0,1]
""" 
def prep_test_images(file_paths,direc,num_images=1000):
    xTest = np.zeros((num_images,224,224,3))
    for i in range(xTest.shape[0]):
        img_path = file_paths[i]
        img_PIL = Image.open(os.path.join(direc,img_path))
        img = np.asarray(img_PIL)
        xTest[i] = img
    xTest = imagenet_utils.preprocess_input(xTest)
    xTest /= 255
    return xTest

""" Given a category name returns the category ID
[ Params ]:
    name: the name of the category
[ Returns ]:
    categoryID: the unique ID of the correct decision
"""
def category_map(name):
    category_names = ['false_detection', 'airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'aquaculture', 'archaeological_site', 'barn', 'border_checkpoint', 'burial_site', 'car_dealership', 'construction_site', 'crop_field', 'dam', 'debris_or_rubble', 'educational_institution', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'flooded_road', 'fountain', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'interchange', 'lake_or_pond', 'lighthouse', 'military_facility', 'multi-unit_residential', 'nuclear_powerplant', 'office_building', 'oil_or_gas_facility', 'park', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'port', 'prison', 'race_track', 'railway_bridge', 'recreational_facility', 'impoverished_settlement', 'road_bridge', 'runway', 'shipyard', 'shopping_mall', 'single-unit_residential', 'smokestack', 'solar_farm', 'space_facility', 'stadium', 'storage_tank','surface_mine', 'swimming_pool', 'toll_booth', 'tower', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
    return category_names.index(name)
  


if __name__ == '__main__':
     main(sys.argv)
