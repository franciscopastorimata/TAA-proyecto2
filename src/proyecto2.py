import os
import numpy as np
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
import pandas as pd
from comet_ml import Experiment
# import IPython.display as Image, display
from IPython.display import Image, display
import tensorflow_addons as tfa
from sklearn.utils import class_weight
import matplotlib.cm as cm


def _parse_image_function_train(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    parsed_features = tf.io.parse_single_example(example_proto, feature_description_train)
    return parsed_features['image'], parsed_features['level_cat']

def _parse_image_function_test(example_proto):
    parsed_features = tf.io.parse_single_example(example_proto, feature_description_test)
    return parsed_features['image'], parsed_features['image_name']

def get_dataset(set='train'):
    if set == 'train':
        tf_record_dataset = tf.data.TFRecordDataset([tfrecords_train])
        dataset = tf_record_dataset.map(_parse_image_function_train)
    elif set == 'validation':
        tf_record_dataset = tf.data.TFRecordDataset([tfrecords_val])
        dataset = tf_record_dataset.map(_parse_image_function_train)
    elif set == 'test':
        tf_record_dataset = tf.data.TFRecordDataset([tfrecords_test])
        dataset = tf_record_dataset.map(_parse_image_function_test)
    else:
        raise ValueError(f'"{set}" is not a valid set, valid sets are "train", "validation" and "test"')
    return dataset

def preprocess_train(serialized_example):
    parsed_example = tf.io.parse_single_example(serialized_example, feature_description_train)
    label = parsed_example['level_cat']

    decoded_img = tf.io.decode_jpeg(parsed_example["image"])
    decoded_img = tf.image.convert_image_dtype(decoded_img, tf.float32) * 255
    decoded_img_resized = tf.image.resize(decoded_img, [img_size, img_size])
    
    return decoded_img_resized, label

def preprocess_test(serialized_example):
    parsed_example = tf.io.parse_single_example(serialized_example, feature_description_test)

    decoded_img = tf.io.decode_jpeg(parsed_example["image"])
    decoded_img = tf.image.convert_image_dtype(decoded_img, tf.float32) * 255
    decoded_img_resized = tf.image.resize(decoded_img, [img_size, img_size])
    
    return decoded_img_resized

def get_ds_batches(ds_train, ds_val, batch_size, train_size=None, val_size=0.2):
  
  if train_size != None:
    val_size = int(train_size*0.2)
    ds_train_batches = ds_train.take(train_size).batch(batch_size).prefetch(1)
    ds_val_batches = ds_val.take(val_size).batch(batch_size).prefetch(1)
    print('Train size:'+str(train_size))
    print('Validation size:'+str(val_size))
  else:
    ds_train_batches = ds_train.batch(batch_size).prefetch(1)
    ds_val_batches = ds_val.batch(batch_size).prefetch(1)
  
  return ds_train_batches, ds_val_batches

def Xception(img_size, batch_normalization=False, data_aug=False, drop=None):
    base_model = tf.keras.applications.Xception(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(img_size, img_size, 3),
        include_top=False,
    )  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    if data_aug:
        # inputs = tf.image.adjust_brightness(inputs, delta=0.2)
        inputs = tf.image.stateless_random_brightness(inputs, max_delta=0.8, seed=(1,2))
    x = tf.keras.applications.xception.preprocess_input(inputs)

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # #probar 1 o 2 capas densas.
    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    if drop!=None:
        x = tf.keras.layers.Dropout(drop)(x)
    x = tf.keras.layers.Dense(2048, activation='relu', kernel_initializer='he_normal')(x)

    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    if drop!=None:
        x = tf.keras.layers.Dropout(drop)(x)
    x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_normal')(x)

    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    if drop!=None:
        x = tf.keras.layers.Dropout(drop)(x)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(x)

    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    if drop!=None:
        x = tf.keras.layers.Dropout(drop)(x)
    x = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)

    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    metric_cohenKappa = tfa.metrics.CohenKappa(num_classes=5, sparse_labels=True, weightage='quadratic', name = 'quadratic_weighted_kappa')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),   #1e-2
        loss='sparse_categorical_crossentropy',
        metrics=[metric_cohenKappa, 'accuracy'])
    return model

def Xception_reg(img_size):
  base_model = tf.keras.applications.Xception(
      weights="imagenet",  # Load weights pre-trained on ImageNet.
      input_shape=(img_size, img_size, 3),
      include_top=False,
  )  # Do not include the ImageNet classifier at the top.

  # Freeze the base_model
  base_model.trainable = False

  # Create new model on top
  inputs = tf.keras.Input(shape=(img_size, img_size, 3))
  # x = data_augmentation(inputs)  # Apply random data augmentation

  x = tf.keras.applications.xception.preprocess_input(inputs)

  # The base model contains batchnorm layers. We want to keep them in inference mode
  # when we unfreeze the base model for fine-tuning, so we make sure that the
  # base_model is running in inference mode here.
  x = base_model(x, training=False)
  x = tf.keras.layers.GlobalAveragePooling2D()(x)

  # #probar 1 o 2 capas densas.
  x = tf.keras.layers.Dense(2048, activation='relu', kernel_initializer='he_normal')(x)
  x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_normal')(x)

  x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(x)

  x = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)

  # x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout
  outputs = tf.keras.layers.Dense(1)(x)
  model = tf.keras.Model(inputs, outputs)
  
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),   #1e-2
      loss='mse',
      metrics=[tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error')])
  return model

def create_experiment(tags=None):
    """ Crea el experimento con los tags pasados como variable y lo devuelve."""
    experiment = Experiment(
        api_key='8KM5gTaM4zSkcTrQ4BtwLOmFo',
        project_name="taa-proyecto2",
        workspace=os.getenv('fede-p'),
        auto_param_logging=False,
    )

    if tags:
        for tag in tags:
            experiment.add_tag(tag)

    return experiment


def log_experiment(experiment:Experiment, params=None, metrics=None):
    """ Logea tanto los parametros como las metricas. """
    if params!=None:
        experiment.log_parameters(params)
    if metrics!=None:
        experiment.log_metric("quadratic_weighted_kappa", metrics)
    
    return True

def plot_cohen_kappa(cmt_exp, ck_train, ck_val):
    '''
    Entrada:
        cmt_exp: experimento comet
        ck_train: Cohen Kappa on training data
        ck_val: Cohen Kappa on validation data
    '''
    ### Registro de Gráficas ###
    plt.figure()
    plt.plot(ck_train,'*-', label='train')
    plt.plot(ck_val,'*-', label='validation')
    plt.xlabel('epoch')
    plt.ylabel('Quadratic Weighted Kappa')
    # plt.ylim(0, 4)
    plt.title('Cohen Kappa')
    plt.legend()
    plt.grid()
    
    cmt_exp.log_figure(figure_name="CK" , figure=plt)
   
    return

def plot_loss(cmt_exp, loss_train, loss_val):
    '''
    Entrada:
        cmt_exp: experimento comet
        loss_train: loss on training data
        loss_val: loss on validation data
    '''
    ### Registro de Gráficas ###
    plt.figure()
    plt.plot(loss_train,'*-', label='train')
    plt.plot(loss_val,'*-', label='validation')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.ylim(0, 4)
    plt.title('loss')
    plt.legend()
    plt.grid()
    
    cmt_exp.log_figure(figure_name="loss" , figure=plt)
   
    return

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    #img = keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))


if __name__ == '__main__':
    name = 'first_model'
    img_size = 512
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    
    folder_path = '../data/retinopatia_entrenamiento/retinopatia_entrenamiento/'
    tfr_folder = os.path.join(folder_path, os.listdir(folder_path)[3])
    tfrecords_train = os.path.join(tfr_folder, os.listdir(tfr_folder)[1])
    tfrecords_val = os.path.join(tfr_folder, os.listdir(tfr_folder)[2])
    tfrecords_test = os.path.join(tfr_folder, os.listdir(tfr_folder)[0])

    feature_description_train = {"image": tf.io.FixedLenFeature([], tf.string),
                             "image_name": tf.io.FixedLenFeature([], tf.string),
                             "eye": tf.io.FixedLenFeature([], tf.int64),
                             "level_cat": tf.io.FixedLenFeature([], tf.int64)}

    feature_description_test = {"image": tf.io.FixedLenFeature([], tf.string),
                                "image_name": tf.io.FixedLenFeature([], tf.string),
                                "eye": tf.io.FixedLenFeature([], tf.int64)}

    # img_size = 512
    tfrecord_ds_train = tf.data.TFRecordDataset([tfrecords_train])
    ds_train = tfrecord_ds_train.map(preprocess_train)

    tfrecord_ds_val = tf.data.TFRecordDataset([tfrecords_val])
    ds_val = tfrecord_ds_val.map(preprocess_train)

    tfrecord_ds_test = tf.data.TFRecordDataset([tfrecords_test])
    ds_test = tfrecord_ds_test.map(preprocess_test)

    train_size = 5000
    batch_size = 8
    ds_train_batches, ds_val_batches = get_ds_batches(ds_train, ds_val, batch_size, train_size=train_size, val_size=0.2)

    df_train = pd.read_csv('../data/retinopatia_entrenamiento/retinopatia_entrenamiento/tf_records_nuestros/labels_train.csv')
    df_val = pd.read_csv('../data/retinopatia_entrenamiento/retinopatia_entrenamiento/tf_records_nuestros/labels_val.csv')

    y_train = np.array(df_train['level'])

    w_class = class_weight.compute_class_weight(class_weight = 'balanced',
                                                classes = np.unique(y_train),
                                                y = y_train)                                                  
    w_class= {i:w_class for i, w_class in enumerate(w_class)}
    print(w_class)

    exp = create_experiment()

    # CLASSIFICATION MODEL
    model = Xception(img_size=512, batch_normalization=True, data_aug=False, drop=0.5)
    history = model.fit(ds_train_batches, epochs=20, validation_data=ds_val_batches, class_weight= w_class,
            callbacks = [tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True),
                        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    plot_cohen_kappa(exp, history.history['quadratic_weighted_kappa'], history.history['val_quadratic_weighted_kappa'])
    plot_loss(exp, history.history['loss'], history.history['val_loss'])

    # REGRESSION MODEL
    # model = Xception_reg(img_size=512)
    # history = model.fit(ds_train_batches, epochs=10, validation_data=ds_val_batches,
    #         callbacks = [tf.keras.callbacks.ModelCheckpoint("model_reg.h5", save_best_only=True),
    #                     tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    # plot_loss(exp, history.history['loss'], history.history['val_loss'])

    # PREDICTIONS
    y_pred = model.predict(ds_test.batch(batch_size).prefetch(1))
    test_levels = np.argmax(y_pred, axis=1)
    
    dataset_test = get_dataset('test')
    test_names = []
    for image, name in dataset_test:
        test_names.append(name.numpy().decode('utf-8'))

    d = {'image': test_names, 'level': test_levels}
    df_test = pd.DataFrame(d)
    df_test.to_csv('dropout_submission.csv', index=False)

    # GRAD-CAM
    




    exp.end()

