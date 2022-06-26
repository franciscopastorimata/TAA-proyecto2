import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from commet_experiment import save_experiment_commet
from read_records import get_train_dataset, get_valid_dataset

def model_example(name, train_set, val_set, w_train):
    pass
    """
    model = tf.keras.models.Sequential([
                                        tf.keras.layers.SimpleRNN(64, return_sequences=True ,dropout=drop, recurrent_dropout=drop),
                                        tf.keras.layers.Dense(1, activation='relu')
                                        ])
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss=root_mean_squared_log_error, optimizer=optimizer, metrics=[mean_absolute_error])
    history = model.fit(ds_train, epochs=5, validation_data=ds_val)

    rmsle, mae = model.evaluate(val_set, verbose=2)
    metrics = {
        'val_rmsle': rmsle,
        'val_mae': mae,
    }
    save_experiment_commet(name, model, historys, history, metrics)
    """

def create_simple_cnn(N=2, input_shape=(28,28, 1), activation='relu', optimizer='sgd', learning_rate= 1e-3, 
                        batch_normalization=False, dropout=None, n_filt_1=64, n_filt_2=32, ker_size=(3,3)):
    model = keras.models.Sequential()
    for n in range(N):
        model.add(keras.layers.Conv2D(filters=n_filt_1, kernel_size=ker_size, input_shape=input_shape, activation='relu'))
        if batch_normalization:
            model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(filters=n_filt_2, kernel_size=ker_size, activation='relu'))
        if batch_normalization:
            model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
        if batch_normalization:
            model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Flatten())
    if batch_normalization:
            model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(128, activation=activation))
    if batch_normalization:
            model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    if optimizer=="sgd":
        optimizer = keras.optimizers.SGD(lr=learning_rate)
    elif optimizer=="adam":
        optimizer = keras.optimizers.Adam(lr=learning_rate)
    
    if dropout!=None:
        model.add(keras.layers.Dropout(dropout))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

if __name__ == '__main__':
    train_dataset = get_train_dataset()
    valid_dataset = get_valid_dataset()
    model = create_simple_cnn(N=2, input_shape=(1024,683,3),learning_rate=5e-4)
    history = model.fit(train_dataset, epochs=10, validation_data=valid_dataset)
    print(history.history)
