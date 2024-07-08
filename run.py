import argparse
from ultralytics import YOLO
import random
import numpy as np
import cv2
from keras import layers
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
import os
from keras import layers
import matplotlib.pyplot as plt
from keras.models import Model
from ultralytics import YOLO
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from bidi.algorithm import get_display
from IPython.display import clear_output
import random
tf.config.set_visible_devices([], '')


def main():
    base_dir = 'Data/mergedData'
    path_train = []
    train_labels = []
    images_name_train = os.listdir(base_dir)
    for name in images_name_train:
        train_labels.append(name.split('_')[0])
        path_train.append(os.path.join(base_dir, name))

    characters = set(char for label in train_labels for char in label)
    characters = sorted(list(characters))
    char_to_num = layers.StringLookup(
        vocabulary=list(characters), mask_token=None
    )
    num_to_char = layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )
    character_mapping = {
        "B": "ب",
        "C": "س",
        "D": "د",
        "H": "ه",
        "J": "ج",
        "K": "ط",
        "L": "ل",
        "M": "م",
        "N": "ن",
        "S": "ص",
        "T": "ت",
        "V": "و",
        "X": "ق",
        "Z": "ع",
        "A": "الف",
        "Y": "ی",
        "P": "پ",
        "F": "معلولین",
        "Q": "ز"
    }




    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    args = parser.parse_args()



    model = tf.keras.models.load_model('Models/ModelTForcingAttention/TForcingAttentionModel.tf')
    model_pose = YOLO('Yolo/best_test2_with_nano/best.pt')
    encoder = model.get_layer('model_1')
    decoder = model.get_layer('model_2')
    characters = set(characters)
    characters.add('G')
    characters.add('E')

    START_TOKEN = char_to_num('G')
    END_TOKEN = char_to_num('E')
    characters = sorted(list(characters))
    char_to_num = layers.StringLookup(
        vocabulary=list(characters), mask_token=None
    )
    num_to_char = layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )



    results = model_pose(args.input)[0]
    test_image = cv2.imread(args.input)

    predicted = []
    test_list = []
    for key, value in enumerate(range(len(results))):
        topLeft = tuple(map(int, results[key].keypoints.data[0][0]))
        topRight = tuple(map(int, results[key].keypoints.data[0][1]))
        bottomRight = tuple(map(int, results[key].keypoints.data[0][2]))
        bottomLeft = tuple(map(int, results[key].keypoints.data[0][3]))
        points = np.array([topLeft, topRight, bottomRight, bottomLeft], np.int32)

        pts1 = []
        pts1.append(topLeft)
        pts1.append(topRight)
        pts1.append(bottomLeft)
        pts1.append(bottomRight)
        pts1 = np.array(pts1, dtype='float32')
        clone = test_image.copy()
        pts2 = np.float32([[0,0],[256,0],[0,65],[256,65]])
        if len(pts1) == 4:
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(clone, M, (256, 65))

            
            resized_image = cv2.resize(dst, (256, 65))
            resized_image = tf.transpose(resized_image, perm=[1, 0, 2])
            resized_image = resized_image[np.newaxis, ...]
            resized_image = tf.cast(resized_image, tf.float32)
            resized_image = resized_image / 255.0


            encoder_outputs, h, c = encoder(resized_image, training=False)
            outputs = tf.expand_dims([START_TOKEN] * 1, axis=1)
            for i in range(9):
                dec_inp = outputs[:, -1]
                dec_inp = dec_inp[:, tf.newaxis]
                output, h, c = decoder([dec_inp, encoder_outputs, h, c], training=False)
                output = tf.argmax(output[:, 0], axis=-1)
                outputs = tf.concat([outputs, output[tf.newaxis, ...]], axis=-1)

            resulting_string = ''.join([num_to_char(x).numpy().decode('utf-8') for x in outputs.numpy()[0]][1:-1])
            predicted.append(resulting_string)
        resulting_string = predicted[key]
        tensor = results[key].boxes.xyxy[0]
        tensor_list = tensor.tolist()
        top_left = (int(tensor_list[0]), int(tensor_list[1]))
        bottom_right = (int(tensor_list[2]), int(tensor_list[3]))
        cv2.rectangle(test_image, top_left, bottom_right, color=(0, 255, 0), thickness=5)
        cv2.polylines(test_image, [points], isClosed=True, color=(0, 0, 255), thickness=2)
        text = str(resulting_string[3:] + character_mapping[resulting_string[2]] + resulting_string[:2])
        
        scale = int(test_image.shape[1] / 1000)
        scale_font = 40 * scale
        font = ImageFont.truetype("fonts/B_Koodak_Bold_0.ttf", size=(40 + scale_font))
        
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_image = Image.fromarray(test_image)
        draw = ImageDraw.Draw(test_image)
        scale_move = scale * 40
        draw.text((int(top_left[0]), int(top_left[1]) - (40 + scale_move)), text, font=font, fill=(0, 255, 0))
        test_image = np.array(test_image)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        print(predicted[key])
    cv2.imwrite('results/output_' + str(random.randint(1, 10000)) + '.jpg', test_image)




























if __name__ == "__main__":
    main()