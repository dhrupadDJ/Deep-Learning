# build_model.py
import tensorflow as tf

def build_model(input_shape, learning_rate=0.0009):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                  metrics=['accuracy'])
    return model
