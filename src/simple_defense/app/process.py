#This is the file to compile and create the model
import sys
import redis
import random
import os
import json
sys.path.append('./data_ml_functions/DenseNet')
from keras.models import Model, load_model, Sequential
import numpy as np
import tensorflow as tf
from keras.applications import imagenet_utils
from keras.utils.np_utils import to_categorical
from PIL import Image
from data_ml_functions.mlFunctions import get_cnn_model, img_metadata_generator,get_lstm_model,codes_metadata_generator
 
def main(args):
    model = Sequential()

    model = load_model('cnn_image_only.model')
    model.compile(loss='categorical_crossentropy', optimizer='SGD',metrics=['accuracy'])
    
    input_path= args[1]
    output_file = args[2]
    print(output_file)
    
    yTest = np.load(os.path.join(input_path,'testlabels.npy'))
    file_list = [0] * yTest.shape[0]
    for val in os.listdir(input_path):
        if val.endswith('.jpg'):
            num = int(val.split('_')[-1][:-4])
            file_list[num] = os.path.join(input_path,val)
        
    xTest = prep_test_images(file_list)
    yTest_unsorted = np.load(os.path.join(input_path,'testlabels.npy'))
    preds = model.predict(xTest)
    np.save(output_file,preds)
    hits = 0
    for pred, y in zip(preds, yTest):
        if np.argmax(pred) == np.argmax(y):
            hits += 1
    print(float(hits)/yTest.shape[0])
    
def prep_test_images(file_paths,num_images=1000):
    xTest = np.zeros((num_images,224,224,3))
    for i in range(xTest.shape[0]):
        img_path = file_paths[i]
        img_PIL = Image.open(img_path)
        img = np.asarray(img_PIL)
        xTest[i] = img
    xTest = imagenet_utils.preprocess_input(xTest)
    xTest /= 255
    return xTest

def save_test_set(filelist, num_images=1000):
    save_dir = 'test_images'
    yTest = np.zeros((num_images, 63))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    for i in range(num_images):
        img_path = filelist[i]
        img_PIL = Image.open(img_path)
        img_PIL.save(os.path.join(save_dir,'testing_'+str(i)+'.jpg'))
        category = category_map(img_path.split('/')[-2])
        yTest[i] = to_categorical(category, 63)

    np.save(os.path.join(save_dir,'testlabels.npy'), yTest)


def prep_filelist(data_dir='train/'):
    all_paths = []
    cache = os.path.join(data_dir, 'cache')
    if not os.path.exists(cache):
        os.mkdir(cache)
    if os.path.exists(os.path.join(cache, 'filepaths.npy')):
        all_paths = np.load(os.path.join(cache, 'filepaths.npy'))
    else:
        folders = os.listdir(data_dir)
        print(folders)
        for folder in folders:
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                img_files = os.listdir(folder_path)
                for img_file in img_files:
                    if img_file.endswith('.jpg'):
                        all_paths.append(os.path.join(folder_path, img_file))
        random.shuffle(all_paths)
        np.save(os.path.join(cache, 'filepaths.npy'), all_paths)
    print("[ INFO ]:  Prepped " + str(len(all_paths)) + " image paths")
    return all_paths

def category_map(name):

    category_names = ['false_detection', 'airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'aquaculture', 'archaeological_site', 'barn', 'border_checkpoint', 'burial_site', 'car_dealership', 'construction_site', 'crop_field', 'dam', 'debris_or_rubble', 'educational_institution', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'flooded_road', 'fountain', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'interchange', 'lake_or_pond', 'lighthouse', 'military_facility', 'multi-unit_residential', 'nuclear_powerplant', 'office_building', 'oil_or_gas_facility', 'park', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'port', 'prison', 'race_track', 'railway_bridge', 'recreational_facility', 'impoverished_settlement', 'road_bridge', 'runway', 'shipyard', 'shopping_mall', 'single-unit_residential', 'smokestack', 'solar_farm', 'space_facility', 'stadium', 'storage_tank','surface_mine', 'swimming_pool', 'toll_booth', 'tower', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
    return category_names.index(name)
  


if __name__ == '__main__':
     main(sys.argv)
