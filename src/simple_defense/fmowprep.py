import os
from keras.utils.np_utils import to_categorical
import json 
import params
import numpy as np
import random
from PIL import Image

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
        contextMultHeight = 0.15):
   filename = os.path.basename(filepath)
    jsonData = json.load(open(filepath))
    img_filename = filepath[:-5] + '.jpg' 
    basename = os.path.basename(img_filename[:-4])
    allResults = []
    allCats = []
    for bb in jsonData['bounding_boxes']:
        category = bb['category']
        box = bb['box']

        outBaseName = basename+'_'+ ('%d' % bb['ID']) + '.jpg'

        cat_dir = os.path.join(train_dir, category+'/')
        currOut = os.path.join(cat_dir, outBaseName)
        if not os.path.exists(currOut):
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
            subImg_PIL = subImg_PIL.resize(params.target_img_size)
            allResults.append(np.asarray(subImg_PIL))
            subImg_PIL.save(currOut)
            cat_value = params.category_names.index(category) 
            allCats.append(to_categorical(cat_value, params.num_labels))
            return np.asarray(allResults), allCats
        else:
            return None, category
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
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    load_from_full(params.directories['dataset'], params.directories['train'])
    #print(image_batch())




