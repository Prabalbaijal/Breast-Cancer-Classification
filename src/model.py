from tensorflow import keras

def create_model(input_dim=30):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(input_dim,)),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(2, activation='sigmoid')   
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
