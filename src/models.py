import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from sklearn.model_selection import GridSearchCV
from read_records import read_tf_training_records, read_tf_val_records, preprocess
import numpy as np
from comet_ml import Experiment
#from commet_experiment import save_experiment_commet
from read_records import get_dataset

def run_first_experiment(name, img_size):
    
    # Se crea un experimento utilizando nuestra API_KEY
    COMET_API_KEY = '8KM5gTaM4zSkcTrQ4BtwLOmFo'
    exp = Experiment(
                      api_key=COMET_API_KEY,
                      project_name="taa-proyecto2",
                      workspace="fede-p")
    exp.set_name(name)

    #Carga de datos y preprocesado
    train_dataset = read_tf_training_records()
    train_dataset = train_dataset.map(preprocess)
    
    valid_dataset = read_tf_val_records()
    valid_dataset = valid_dataset.map(preprocess)

    num_of_elements = 35126
    if img_size == 512:
      batch_size=16
    elif img_size == 256 or img_size == 384:
      batch_size=32
    elif img_size == 128:
      batch_size=64
    
    ds_train = train_dataset.batch(batch_size).prefetch(1)
    ds_val = valid_dataset.batch(batch_size).prefetch(1)

    #Modelo
    base_model = tf.keras.applications.Xception(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(img_size, img_size, 3),
        include_top=False,
    )  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = True

    # Create new model on top
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))

    data_augmentation = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
                                         tf.keras.layers.experimental.preprocessing.RandomRotation(0.1)])

    x = data_augmentation(inputs)  # Apply random data augmentation

    # Pre-trained Xception weights requires that input be scaled
    # from (0, 255) to a range of (-1., +1.), the rescaling layer
    # outputs: `(inputs * scale) + offset`
    scale_layer = tf.keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    x = scale_layer(x)

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(x, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = tf.keras.layers.Dense(5)(x)
    model = tf.keras.Model(inputs, outputs)

    metric = tfa.metrics.CohenKappa(num_classes=5, sparse_labels=False, weightage='quadratic')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[metric, 'accuracy'])
   
    #model.summary()
    #Entrenamiento
    
    history = model.fit(ds_train, validation_data=ds_val, epochs = 6)

    '''  
    # Se realiza la busqueda de hiperparámetros
    print(searcher.best_params_)
    exp.log_parameters(params)
    exp.log_parameters(searcher.best_params_, prefix='best')
    save_gs_results(exp, searcher) 
    '''   
    
    # Se guarda el modelo
    #best_estimator = searcher.best_estimator_
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_path = os.path.join('./../saved_models/first_model',name+'.h5')
    tf.keras.models.save_model(model, model_path)
    exp.log_model('retino', model_path)
    
    exp.end()

if __name__ == '__main__':
    name = 'first_model'
    img_size = 256
    run_first_experiment(name = name+'-img_size='+str(img_size), img_size = img_size)
    train_dataset = get_dataset('train')
    valid_dataset = get_dataset('validation')
    model = create_simple_cnn(N=2, input_shape=(1024,683,3),learning_rate=5e-4)
    history = model.fit(train_dataset, epochs=10, validation_data=valid_dataset)
    print(history.history)
