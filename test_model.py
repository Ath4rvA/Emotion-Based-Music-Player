from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten
from keras.layers import Dense, Dropout


def model():
    from keras.preprocessing.image import ImageDataGenerator

    x_train_generator = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    x_val_generator = ImageDataGenerator(rescale=1. / 255)

    x_train_dir = 'images/DATA/train'
    x_val_dir = 'images/DATA/val'

    x_train = x_train_generator.flow_from_directory(x_train_dir,
                                                    target_size=(48, 48),
                                                    color_mode='grayscale',
                                                    batch_size=32,
                                                    class_mode='categorical')

    x_val = x_val_generator.flow_from_directory(x_val_dir,
                                                target_size=(48, 48),
                                                color_mode='grayscale',
                                                batch_size=32,
                                                class_mode='categorical')

    # CNN
    model = Sequential()

    # 1st Block
    model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='elu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='elu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # 2nd Block
    model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # 3rd Block
    model.add(Conv2D(128, (3, 3), activation='elu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='elu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())


    # Fully Connected Network
    model.add(Dense(256, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(7, activation='softmax'))

    # print  Model Summary
    print(model.summary())

    # compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd', metrics=['accuracy'])

    # fit the model
    history = model.fit_generator(x_train,
                        steps_per_epoch=5000,
                        epochs=10,
                        validation_data=x_val,
                        validation_steps=1000)


    json = model.to_json()
    file = open('my_model.json', 'w')
    file.write(json)
    file.close()

    # save the weights
    model.save_weights('weights.h5', True)

model()
