from tensorflow.keras.layers import LeakyReLU
import numpy as np
import tensorflow as tf                     # <--- CRITICAL FIX INCLUDED HERE
from tensorflow.keras.models import Sequential # <--- CRITICAL FIX INCLUDED HERE
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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



def custom_grid_search_for_rnn(config_dict, X_data, y_data, df_train, permutation_id, total_runs, rnn_type='SimpleRNN'):
    """
    Executes a custom forward-chaining grid search fold routine.
    Leverages df_train metadata to reverse rolling Z-score normalization dynamically 
    per fold, evaluating Close Price using true unscaled RMSE and Theil's U Statistic.
    """
    # Force this specific process to initialize its own clean, isolated CPU thread
    tf.config.set_visible_devices([], 'GPU') 

    print(f"-> Starting Job [{permutation_id}/{total_runs}]: {config_dict['rnn_units']} | Batch: {config_dict['batch_size']}")

    tscv = TimeSeriesSplit(n_splits=3)
    fold_val_losses = []
    fold_unscaled_rmses = []
    fold_theils_u = []

    # Execute chronological forward-chaining splits
    for train_idx, val_idx in tscv.split(X_data):
        X_tr, X_val = X_data[train_idx], X_data[val_idx]
        y_tr, y_val = y_data[train_idx], y_data[val_idx]

        # Instantiate model from custom global wrapper function
        # Dynamically matches features input size (e.g., shape of index 2 of your matrix)
        model = create_rnn_model(
            input_shape=(21, X_data.shape[2]),
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

        model.compile(optimizer=opt, loss=config_dict['loss_function'])

        # Tightened parameters to catch local validation minimums and mitigate drift
        lrReducerOnPlateau = ReduceLROnPlateau(
            monitor='val_loss', 
            mode='min', 
            factor=config_dict['lr_factor'], 
            patience=config_dict['lr_patience'], 
            min_delta=0
        )
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train strictly on this worker's assigned internal thread
        with tf.device('/CPU:0'):
            model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=40,
                batch_size=config_dict['batch_size'],
                callbacks=[lrReducerOnPlateau, early_stop],
                verbose=0,
                shuffle=False
            )

        # 1. Track optimal validation loss (Corresponds to your native training scale, e.g., MAE)
        best_loss = min(model.history.history['val_loss'])
        fold_val_losses.append(best_loss)

        # 2. Extract scaled predictions on validation slice
        preds_scaled = model.predict(X_val, verbose=0).flatten()
        y_val_flat = y_val.flatten()

        # 3. CRUCIAL UNIPOTENT INDEX FIX: 
        # Extract the exact rolling mean and std mapped to this validation fold's true chronological row indices
        meta_slice = df_train.iloc[val_idx]
        val_means = meta_slice["y_mean"].values
        val_stds = meta_slice["y_std"].values

        # 4. REVERSE WINDOW SCALING: Manually convert values back to original unscaled dollar amounts
        preds_original = (preds_scaled * (val_stds + 1e-8)) + val_means
        y_val_original = (y_val_flat * (val_stds + 1e-8)) + val_means

        # 5. METRIC EVALUATION A: Calculate true, uninflated RMSE in real currency units
        unscaled_rmse = np.sqrt(np.mean((y_val_original - preds_original) ** 2))
        fold_unscaled_rmses.append(unscaled_rmse)

        # 6. METRIC EVALUATION B: Calculate Theil's U Statistic to benchmark against a random walk
        # Naive forecast assumption: Tomorrow's Close Price equals Today's Close Price
        naive_preds = y_val_original[:-1]
        actual_today = y_val_original[1:]
        model_preds_today = preds_original[1:]

        model_rmse_horizon = np.sqrt(np.mean((actual_today - model_preds_today) ** 2))
        naive_rmse_horizon = np.sqrt(np.mean((actual_today - naive_preds) ** 2))

        theils_u = model_rmse_horizon / (naive_rmse_horizon + 1e-8)
        fold_theils_u.append(theils_u)

        # Package unified record including your new multi-metric evaluations
    result_record = {**config_dict} 
    result_record["Mean Val MAE"] = round(np.mean(fold_val_losses), 4)
    result_record["Mean Unscaled RMSE"] = round(np.mean(fold_unscaled_rmses), 4)
    result_record["Mean Theils U"] = round(np.mean(fold_theils_u), 4)

    # ERROR WAS HERE: Changed 'Mean Unscaled_RMSE' to 'Mean Unscaled RMSE'
    print(f"<= Completed Job [{permutation_id}/{total_runs}]: MAE Loss = {result_record['Mean Val MAE']} | Unscaled RMSE = ${result_record['Mean Unscaled RMSE']} | Theil's U = {result_record['Mean Theils U']}")
    return result_record
