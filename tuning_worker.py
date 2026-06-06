from tensorflow.keras.layers import LeakyReLU
import numpy as np
import tensorflow as tf                     # <--- CRITICAL FIX INCLUDED HERE
from tensorflow.keras.models import Sequential # <--- CRITICAL FIX INCLUDED HERE
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit

# Create a function that creates a simple RNN model according to the model configuration arguments
def create_rnn_model(input_shape, rnn_type='LSTM', rnn_units=64, dense_units=32, dropout_rate=0.2, internal_dropout=None, internal_activation=None, activation_fn=None):
    model = Sequential()

    model_params = {'units': rnn_units, 'return_sequences': False, 'input_shape': input_shape}

    if internal_dropout is not None:
        model_params['dropout'] = internal_dropout

    if internal_activation is not None:
        model_params['recurrent_activation'] = internal_activation

    if rnn_type == 'LSTM':
        model.add(LSTM(**model_params))
    elif rnn_type == 'GRU':
        model.add(GRU(**model_params))
    elif rnn_type == 'SimpleRNN':
        model.add(SimpleRNN(**model_params))
    else:
        raise ValueError("Invalid RNN type. Choose from 'LSTM', 'GRU', or 'SimpleRNN'.")

    if activation_fn is not None:
        if isinstance(activation_fn, str):
            # If it is a string name like 'relu' or 'tanh'
            model.add(Dense(dense_units, activation=activation_fn))
        else:
            # FIX: If it is an advanced layer instance like LeakyReLU(alpha=0.1),
            # add the Dense layer unactivated first, then append the layer instance.
            model.add(Dense(dense_units))
            model.add(activation_fn)
    else:
        model.add(Dense(dense_units, activation='relu'))


    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(Dense(1))  # Output layer for regression
    return model



def custom_grid_search_for_rnn(config_dict, X_data, y_data, permutation_id, total_runs, rnn_type='SimpleRNN'):
    # CRITICAL FIX: Force this specific process to initialize its own clean, 
    # isolated virtual CPU device thread so it never conflicts with other concurrent processes.
    tf.config.set_visible_devices([], 'GPU') # Turn off GPU allocation for this process

    print(f"-> Starting Job [{permutation_id}/{total_runs}]: {config_dict['rnn_units']} | Batch: {config_dict['batch_size']}")

    tscv = TimeSeriesSplit(n_splits=3)
    fold_val_losses = []
    fold_hit_rates = []

    # Execute chronological forward-chaining folds
    for train_idx, val_idx in tscv.split(X_data):
        X_tr, X_val = X_data[train_idx], X_data[val_idx]
        y_tr, y_val = y_data[train_idx], y_data[val_idx]

        # Instantiate model from global wrapper function
        # We unpack using local keys to match your custom wrapper function signature
        model = create_rnn_model(
            input_shape=(21,5),
            rnn_type=rnn_type,
            rnn_units=config_dict['rnn_units'],
            dense_units=config_dict['dense_units'],
            dropout_rate=config_dict['dropout_rate'],
            internal_dropout=config_dict['internal_dropout'],
            activation_fn=config_dict['activation_fn']
        )

        # Map optimizer
        if config_dict['optimizer_name'] == 'Adam':
            opt = Adam(learning_rate=config_dict['learning_rate'])
        else:
            opt = SGD(learning_rate=config_dict['learning_rate'])

        model.compile(optimizer=opt, loss='mse')

        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

        # Train strictly on this worker's assigned internal thread
        with tf.device('/CPU:0'):
            model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=20,
                batch_size=config_dict['batch_size'],
                callbacks=[early_stop],
                verbose=0,
                shuffle=False
            )

        # 1. Evaluate distance performance (MSE)
        best_loss = min(model.history.history['val_loss'])
        fold_val_losses.append(best_loss)

        # 2. Evaluate directional prediction performance (Directional Hit Rate)
        preds = model.predict(X_val, verbose=0).flatten()
        hit_rate = np.mean(np.sign(y_val) == np.sign(preds))
        fold_hit_rates.append(hit_rate)

        # Package final combined results for this specific parameter set
    result_record = {**config_dict} 
    result_record["Mean Val MSE"] = round(np.mean(fold_val_losses), 4)
    # Store directional metrics as a float for sorting purposes later
    result_record["Mean Hit Rate"] = np.mean(fold_hit_rates)

    print(f"<= Completed Job [{permutation_id}/{total_runs}]: MSE = {result_record['Mean Val MSE']} | Hit Rate = {result_record['Mean Hit Rate']:.2%}")
    return result_record
