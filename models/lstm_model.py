import tensorflow as tf


def build_lstm_model(input_length: int,
                     units: int = 64,
                     learning_rate: float = 1e-3) -> tf.keras.Model:

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_length, 1)),
        tf.keras.layers.LSTM(units, return_sequences=False),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )

    return model


def train_lstm_model(X_train, y_train, X_val=None, y_val=None,
                     epochs=40, batch_size=32, input_length=None):

    if input_length is None:
        input_length = X_train.shape[1]

    model = build_lstm_model(input_length=input_length)

    callbacks = []
    if X_val is not None and y_val is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )
        )

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
    else:
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

    return model
