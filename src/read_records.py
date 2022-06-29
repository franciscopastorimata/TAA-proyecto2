import tensorflow as tf
import os

TRAIN_FULL_TF_RECORDS_PATH = "./../data/retinopatia_entrenamiento/tf_records"
TRAIN_TF_RECORDS_PATH = "./../data/retinopatia_entrenamiento/images_train.tfrec"
VAL_TF_RECORDS_PATH = "./../data/retinopatia_entrenamiento/images_val.tfrec"
TEST_TF_RECORDS_PATH = ""

# Create a dictionary describing the features.
image_feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'image_name': tf.io.FixedLenFeature([], tf.string),
    'eye': tf.io.FixedLenFeature([], tf.int64),
    'level_cat': tf.io.FixedLenFeature([], tf.int64),
}

def read_full_tf_training_records():
    return tf.data.TFRecordDataset(
        [f"{TRAIN_FULL_TF_RECORDS_PATH}/{tf_record_name}"
         for tf_record_name in os.listdir(TRAIN_FULL_TF_RECORDS_PATH)])

def read_tf_training_records():
    return tf.data.TFRecordDataset(
        [f"{TRAIN_TF_RECORDS_PATH}"])

def read_tf_val_records():
    return tf.data.TFRecordDataset(
        [f"{VAL_TF_RECORDS_PATH}"])

def read_tf_test_records():
    return tf.data.TFRecordDataset(
        [f"{TEST_TF_RECORDS_PATH}"])

def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
    parsed_features = tf.io.parse_single_example(example_proto, image_feature_description)
    return parsed_features['image'], parsed_features['level_cat']
  
def get_train_dataset():
    train_dataset = read_tf_training_records()
    parsed_image_dataset = train_dataset.map(_parse_image_function)
    return parsed_image_dataset

def get_valid_dataset():
    val_dataset = read_tf_training_records()
    parsed_image_dataset = val_dataset.map(_parse_image_function)
    return parsed_image_dataset

def preprocess(serialized_example):
    img_size = 256
    parsed_example = tf.io.parse_single_example(serialized_example, image_feature_description)
    label = parsed_example['level_cat']
    decoded_img = tf.io.decode_jpeg(parsed_example["image"])
    decoded_img = tf.image.convert_image_dtype(decoded_img, tf.float32)
    decoded_img_resized = tf.image.resize(decoded_img, [img_size, img_size])
  
    return decoded_img_resized, tf.one_hot(label, depth=5)

if __name__ == '__main__':
    pass
