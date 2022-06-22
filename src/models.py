import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from commet_experiment import save_experiment_commet

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
