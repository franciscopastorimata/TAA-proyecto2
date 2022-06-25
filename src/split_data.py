import tensorflow as tf
import numpy as np
import pandas as pd
import os
import shutil
from read_records import read_full_tf_training_records
from sklearn.model_selection import train_test_split

def split_train_val():
    labels = "../data/retinopatia_entrenamiento/trainLabels.csv"
    train_dir = "../data/retinopatia_entrenamiento/train/"
    val_dir = "../data/retinopatia_entrenamiento/val/"
    os.makedirs(val_dir, exist_ok=True)

    data = pd.read_csv(labels)
    data_left = data.iloc[::2].reset_index(drop=True)
    data_right = data.iloc[1::2].reset_index(drop=True)
    left_train, left_val, y_train, y_val = train_test_split(data_left['image'], data_left['level'], test_size=0.15, random_state=42, stratify=data_left['level'])

    for i in range(len(left_val)):
        file_name_left = list(left_val)[i] + '.jpeg'
        file_name_right = file_name_left.replace('left', 'right')

        left_path_src = os.path.join(train_dir, file_name_left)
        right_path_src = os.path.join(train_dir, file_name_right)
        
        left_path_dst = os.path.join(val_dir, file_name_left)
        right_path_dst = os.path.join(val_dir, file_name_right)
        
        shutil.move(left_path_src, left_path_dst)
        shutil.move(right_path_src, right_path_dst)
    
    data_val = pd.concat([data_left.iloc[list(left_val.index)], data_right.iloc[list(left_val.index)]])
    data_val.sort_values("image", inplace=True)

    data_train = pd.concat([data_left.iloc[list(left_train.index)], data_right.iloc[list(left_train.index)]])
    data_train.sort_values("image", inplace=True)
    
    labels_val = "../data/retinopatia_entrenamiento/labels_val.csv"
    data_val.to_csv(labels_val)

    labels_train = "../data/retinopatia_entrenamiento/labels_train.csv"
    data_train.to_csv(labels_train)

    return data_train, data_val




# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_string, label, filename, eye):
    image_shape = tf.io.decode_jpeg(image_string).shape
    feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'image_name': _bytes_feature(filename.encode()),
      'eye': _int64_feature(eye),
      'level_cat': _int64_feature(label),
      'image': _bytes_feature(image_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def write_training_val_tf_files(df, folder='train'):
    record_file = f'../data/retinopatia_entrenamiento/images_{folder}.tfrec'

    with tf.io.TFRecordWriter(record_file) as writer:
        for row in df.index:
            full_path = f'../data/retinopatia_entrenamiento/{folder}/' + df['image'][row] + '.jpeg'
            label = df['level'][row]
            image_string = tf.io.read_file(full_path)
            filename = df['image'][row]
            eye = filename.split('_')[1] == 'left'
            tf_example = image_example(image_string, label, filename, eye)
            writer.write(tf_example.SerializeToString())

    return

if __name__ == '__main__':
    input_dir = "../data/retinopatia_entrenamiento/"
    if 'val' not in os.listdir(input_dir):
        df_train, df_val = split_train_val()
    else:
        df_train = pd.read_csv("../data/retinopatia_entrenamiento/labels_train.csv")
        df_val = pd.read_csv("../data/retinopatia_entrenamiento/labels_val.csv")

    write_training_val_tf_files(df_train, folder='train')
    write_training_val_tf_files(df_val, folder='val')
