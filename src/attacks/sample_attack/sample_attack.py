import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
import numpy as np
from PIL import Image
import params
import random
import pdb

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, load_model
from keras.applications import imagenet_utils
from keras.utils.np_utils import to_categorical
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod

def main():
    ### Params of the data are here
    data_dir = '/home/neilf/Fendley/adversarial/ISC_AML_2018/image_sets/val_prepped'
    adv_fgsm(data_dir)
   

"""
Attacks the fmow baseline model with an FGSM attack from cleverhans
[  Params  ]: 
    data_dir: directory that stores the image files
    num_adv: the number of adversarial examples to generate, 'max' is all
"""
def adv_fgsm(data_dir,num_adv='max'):
    sess = tf.Session()
    K.set_learning_phase(0)
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())

    model = Sequential()
    model = load_model('cnn_image_only.model')
    model.compile(loss='categorical_crossentropy', optimizer='SGD',metrics=['accuracy'])
    
    ### This is ineffiecient, will speed up 
    img_paths = prep_filelist(data_dir)
    xTest, yTest = prep_adv_set(model,img_paths,num_adv=num_adv)

    x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(None, 63))

    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap, sess=sess)

    # Generate perturbations at various epsilon
    eps = [0.0,0.005,0.01, 0.03, 0.04, 0.1]
    save_folder = 'adv_out/'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    for ep in eps:
        fgsm_params = {'eps': ep}
        adv_x = fgsm.generate(x, **fgsm_params)
        img_adv_out = np.zeros((0,224,224,3),dtype=np.uint8)
        eval_par = {'batch_size': 32}
        counter = 0
        hits = 0

        for i in range(len(xTest)):
            img_clean = np.expand_dims(xTest[i], axis=0)
            if ep == 0:
                img = img_clean
            else:
                img = adv_x.eval(session=sess, feed_dict={x:img_clean})
            cat = yTest[i]
            cat_input = np.expand_dims(cat, axis=0)
            pred = model.predict(img, batch_size=1)
            img_out = ((img + 1) * 255).astype(np.uint8)

            img_adv_out = np.concatenate((img_adv_out,img_out), axis =0)
            if np.argmax(pred) == np.argmax(cat_input):
                hits += 1
            counter += 1
        
        print(img_adv_out.shape)
        for i in range(img_adv_out.shape[0]):
            img_PIL = Image.fromarray(img_adv_out[i])
            img_PIL.save(os.path.join(save_folder,"adv_img_"+str(ep)+
                "_"+str(i)+".png"))
        print("[  Info  ]: The accuracy on eps " + str(ep) + ': ' +str(float(hits)/counter))

"""
This will return a list of all the paths to the png files
[  Params  ]:
    data_dir: directory of the pngs
[  Returns ]:
    file_paths: list of paths to all of the png files
"""
def prep_filelist(data_dir):
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
 

def prep_adv_set(model, filepaths, num_adv=1000):
    if num_adv == 'max':
        num_adv = len(filepaths)
    random.shuffle(filepaths)
    i = 0
    x_test_final = np.zeros((0,224,224,3))
    y_test_final = np.zeros((0,63))
    print("[  INFO  ]:  We are loading correctly detected images")
    batch_size = 512
    while x_test_final.shape[0] < num_adv:
        x_batch = np.zeros((batch_size,224,224,3))
        y_batch = np.zeros((batch_size,63))
        for j in range(batch_size):
            img_path = filepaths[i]
            i += 1
            img_pil = Image.open(img_path)
            x_test = np.asarray(img_pil).astype(np.float32)
            category = img_path.split('/')[-2]
            x_test = imagenet_utils.preprocess_input(x_test)
            x_test = x_test / 255.0
            y_test = to_categorical(params.category_names.index(category), params.num_labels)
            x_batch[j] = x_test
            y_batch[j] = y_test
        print("[  STATUS  ] Predicting image", i, "with batch size", batch_size)
        pred_batch = model.predict(x_batch, batch_size=batch_size)
        for idx in range(batch_size):
            if np.argmax(y_batch[idx]) == np.argmax(pred_batch[idx]):
                x_test_final = np.concatenate((x_test_final, np.expand_dims(x_batch[idx],axis=0)), axis = 0)
                y_test_final = np.concatenate((y_test_final, np.expand_dims(y_batch[idx],axis=0)), axis = 0)

        print ("[  STATUS  ]:  On image " + str(i) + ', ' + str(x_test_final.shape[0]) + " good images loaded")

    print ("[  Info   ]: Finished loading good clean examples, found ", x_test_final.shape[0], "correct detections in ", i, "images")
    return xTest, yTest



if __name__ =='__main__':
    main()