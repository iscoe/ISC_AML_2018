import pdb
import os
from keras.utils.np_utils import to_categorical
import json 
import numpy as np
import csv
import random
from PIL import Image
from keras.models import load_model
Image.MAX_IMAGE_PIXELS = None


CATEGORY_NAMES = ['false_detection', 'airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'aquaculture', 'archaeological_site', 'barn', 'border_checkpoint', 'burial_site', 'car_dealership', 
'construction_site', 'crop_field', 'dam', 'debris_or_rubble', 'educational_institution', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'flooded_road', 'fountain', 'gas_station', 
'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'interchange', 'lake_or_pond', 'lighthouse', 'military_facility', 'multi-unit_residential', 'nuclear_powerplant', 'office_building', 
'oil_or_gas_facility', 'park', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'port', 'prison', 'race_track', 'railway_bridge', 'recreational_facility', 'impoverished_settlement', 
'road_bridge', 'runway', 'shipyard', 'shopping_mall', 'single-unit_residential', 'smokestack', 'solar_farm', 'space_facility', 'stadium', 'storage_tank','surface_mine', 'swimming_pool', 'toll_booth', 'tower',
'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
    

"""
Function to generate and save a downsized image from original FMOW dataset
[ Params ]: 
    filepath: path to the json file for parsing
    train_dir: directory to save the images parsed from the json file
    contextMultWidth: how much context width to add from the original FMOW image
    contextMultHeight: how much context height to add to the original FMow image
[ Returns ]: 
    allResults: an np.array of the images specified from the json file
    allCats: an np.array of the corresponding categories
"""              
def prep_image(filepath, train_dir, contextMultWidth = 0.15,
        contextMultHeight = 0.15, save=True):
    filename = os.path.basename(filepath)
    jsonData = json.load(open(filepath))
    img_filename = filepath[:-5] + '.jpg' 
    basename = os.path.basename(img_filename[:-4])
    allResults = []
    allCats = []
    for bb in jsonData['bounding_boxes']:
        category = bb['category']
        box = bb['box']

        outBaseName = basename+'_'+ ('%d' % bb['ID']) + '.png'

        cat_dir = os.path.join(train_dir, category+'/')
        currOut = os.path.join(cat_dir, outBaseName)
        if not os.path.exists(currOut):
            if save:
                if not os.path.exists(cat_dir):
                    if not os.path.exists(train_dir):
                        os.mkdir(train_dir)
                    os.mkdir(cat_dir)
            img_pil = Image.open(img_filename)
            img = np.asarray(img_pil)
    
            imgPath = os.path.join(currOut, img_filename)
            
            # train with context around box
           
            wRatio = float(box[2]) / img.shape[0]
            hRatio = float(box[3]) / img.shape[1]
            
            if wRatio < 0.5 and wRatio >= 0.4:
                contextMultWidth = 0.2
            if wRatio < 0.4 and wRatio >= 0.3:
                contextMultWidth = 0.3
            if wRatio < 0.3 and wRatio >= 0.2:
                contextMultWidth = 0.5
            if wRatio < 0.2 and wRatio >= 0.1:
                contextMultWidth = 1
            if wRatio < 0.1:
                contextMultWidth = 2
                
            if hRatio < 0.5 and hRatio >= 0.4:
                contextMultHeight = 0.2
            if hRatio < 0.4 and hRatio >= 0.3:
                contextMultHeight = 0.3
            if hRatio < 0.3 and hRatio >= 0.2:
                contextMultHeight = 0.5
            if hRatio < 0.2 and hRatio >= 0.1:
                contextMultHeight = 1
            if hRatio < 0.1:
                contextMultHeight = 2
            
            
            widthBuffer = int((box[2] * contextMultWidth) / 2.0)
            heightBuffer = int((box[3] * contextMultHeight) / 2.0)

            r1 = box[1] - heightBuffer
            r2 = box[1] + box[3] + heightBuffer
            c1 = box[0] - widthBuffer
            c2 = box[0] + box[2] + widthBuffer

            if r1 < 0:
                r1 = 0
            if r2 > img.shape[0]:
                r2 = img.shape[0]
            if c1 < 0:
                c1 = 0
            if c2 > img.shape[1]:
                c2 = img.shape[1]

            if r1 >= r2 or c1 >= c2:
                continue

            # Here is where we save the image and prepare the results
            subImg = img[r1:r2, c1:c2, :]
            subImg_PIL = Image.fromarray(subImg)
            subImg_PIL = subImg_PIL.resize((224,224))
            allResults.append(np.asarray(subImg_PIL))
            if save:
                print("saving out the image")
                subImg_PIL.save(currOut)
            cat_value = CATEGORY_NAMES.index(category) 
            allCats.append(to_categorical(cat_value, 63))
            return np.asarray(allResults), allCats
        else:
            return np.asarray(Image.open(currOut)), category
"""
Loads data from a dataset outlay in the format specified by the fmow dataset. Will
not save images if fmow_all_filenames.npy file already exists in train_dir.
Data layout expected is as followed.
    Dataset_dir/
        category_1_dir/
            category_1_seq_1_dir/
                category_1_seq_1_img_1.jpg
                category_1_seq_1_1_img_1.json
                category_1_seq_1_img_2.jpg
                category_1_seq_1_1_img_2.json
                ...
            category_1_seq_2_dir/
                ...
            ...
        cat_2_dir
            cat_2_seq_1_dir/
            ...
    [ Params ]: 
        data_dir: the location of the fmow dataset on local linux machine
        train_dir: directory to save out sub images for input to algorithms
    [ Returns ]: 
        all_jsons: a list of all the paths to the files for easy reaccess
"""
def load_from_full(data_dir,train_dir='train/'):
    counter = 0
    if not os.path.exists(os.path.join(train_dir,'fmow_all_filenames.npy')):
        all_jsons = []
        counter = 0
        cats = os.listdir(data_dir)
        for cat in cats:
            cat_folder = os.path.join(data_dir,cat)
            folders = os.listdir(cat_folder)
            for folder in folders:
                direc = os.path.join(cat_folder,folder)
                for filename in os.listdir(direc):
                    if filename.endswith('.json'):
                        final_path = os.path.join(direc,filename)
                        all_jsons.append(final_path)
                        if counter % 1000 == 0:
                            print("[ INFO ]:  Processing image ", counter)
                        counter += 1
                        x,y = prep_image(final_path,train_dir)
        all_jsons = np.asarray(all_jsons)
        random.shuffle(all_jsons)
        np.save((os.path.join(train_dir,'fmow_all_filenames.npy')), all_jsons)
    else:
        all_jsons = np.load(os.path.join(train_dir,'fmow_all_filenames.npy'))
    return all_jsons

def found_all_classes(found):
    for cat in found.keys():
        if not found[cat]:
            return False
    return True

def prep_NN_image(image):
    img = image.copy().astype(np.float32)
    # img_pil = Image.open(image_path)
    # img_pil = img_pil.resize((224, 224), Image.ANTIALIAS)
    # img = np.array(img_pil, dtype=np.float32)
    mean = [103.939, 116.779, 123.68]

    img = img[..., ::-1]

    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]

    img /= 255.0
    return img

def load_good_detections(data_dir, train_dir='train/',num_images=100, load_ms=False):
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    found = {}
    model = load_model('cnn_image_only.model')
    model.compile(loss='categorical_crossentropy', optimizer='SGD',metrics=['accuracy'])
    #CREATE FULL IMAGE LIST WITH CLASS
    data_list = []
    cats = os.listdir(data_dir)
    for cat in cats:
        if cat not in found.keys():
            found[cat] = False
        cat_folder = os.path.join(data_dir,cat)
        sequences = os.listdir(cat_folder)
        for seq in sequences:
            seq_folder = os.path.join(cat_folder, seq)
            for filename in os.listdir(seq_folder):
                if filename.endswith('.json'):
                    if load_ms:
                        final_path = os.path.join(seq_folder, filename)
                        data_list.append((final_path, cat))
                    else:
                        if not filename[:-5].endswith('msrgb'):
                            final_path = os.path.join(seq_folder, filename)
                            data_list.append((final_path, cat))


    random.shuffle(data_list)
    good_images = []
    good_labels = []
    i = 0
    while i < len(data_list) and len(good_images) < num_images:
        json_path = data_list[i][0]
        img_path = json_path[:-5] + '.png'
        img_name = os.path.basename(img_path)
        cat = data_list[i][1]
        x, y = prep_image(json_path, train_dir, save=False)
        x_input = np.expand_dims(prep_NN_image(x[0]), axis=0)
        pred = model.predict(x_input)
        if np.argmax(pred) == np.argmax(y):
            good_images.append(x_input)
            good_labels.append((img_name,np.argmax(y)))
            out_image = Image.fromarray(x[0])
            out_image.save(os.path.join(train_dir, img_name))
        if i % 100 == 0:
            print("We are on %s with accuracy %s" % (i,float(len(good_images)/(i+1))))
        i += 1
    csv_file_name = os.path.join(train_dir, 'labels.csv')
    with open(csv_file_name, 'w') as output:
        writer = csv.writer(output, lineterminator='\n')
        for i in range(len(good_labels)):
            writer.writerow([good_labels[i][0],good_labels[i][1]])


if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    #load_good_detections('/home/fendlnm1/fmow_train_set')
    load_from_full('/home/fendlnm1/fmow_train_set')
    #print(image_batch())




