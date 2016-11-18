from keras.models import Sequential
from keras.layers.core import Flatten, Dropout, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D


def make_model(nb_classes):
    '''make the model via Keras
    
    
    '''
    model = Sequential()

    # Convolution2D(number_filters, row_size, column_size, input_shape=(number_channels, img_row, img_col))
    
    model.add(Convolution2D(6, 5, 5, input_shape=(256, 256, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(16, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(120, 5, 5))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(84))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])
    
    return model
    
def train_model(model, (train_data, train_labels), (val_data, val_labels), nb_epoch=10, batch_size=30):
    '''
    '''
    model.fit(train_data, train_label, batch_size=batch_size, nb_epoch=np_epoch,
              verbose=1, validation_data=(val_data, val_label))
    score = model.evaluate(val_data, val_labels, verbose=1)