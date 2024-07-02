import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil

def preprocess_data():
    if 'validation' not in os.listdir('../Plate_license'):
        os.mkdir('validation')
    if 'test' not in os.listdir('../Plate_license'):
        os.mkdir('test')
    if 'train' not in os.listdir('../Plate_license'):
        os.mkdir('train')
    
    base_dir = './data'
    list_data = os.listdir(base_dir)
    labels = []
    for lbl in list_data:
        labels.append(lbl.split('_')[0])
    
    
    valid_and_test = 0.1
    number_of_validation = int(len(list_data) * valid_and_test)
    number_of_test = int(len(list_data) * valid_and_test)
    number_of_train = len(list_data) - (number_of_test + number_of_validation)
    
    for data in list_data:
        if number_of_validation != 0:
            number_of_validation -= 1
            # move data to validation directory
            shutil.move(os.path.join(base_dir, data), './validation')
        else:
            if number_of_test != 0:
                number_of_test -= 1
                # move data to test directory
                shutil.move(os.path.join(base_dir, data), './test')
            else:
                if number_of_train != 0:
                    number_of_train -= 1
                    # move data to train directory
                    shutil.move(os.path.join(base_dir, data), './train')