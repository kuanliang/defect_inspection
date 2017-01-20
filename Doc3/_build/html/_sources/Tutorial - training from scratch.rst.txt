Tutorial - training from scratch
================================

.. code:: ipython2

    from defectlib import load_tensors
    from defectlib import Config
    from matplotlib import image
    from IPython.display import Image
    import matplotlib.pyplot as plt
    import defectlib
    import cv2
    import numpy as np
    %matplotlib inline


.. parsed-literal::

    Using TensorFlow backend.


.. code:: ipython2

    Image(filename='./images/defect_overview.png')




.. image:: images/output_1_0.png



**原始檔案路徑**

.. code:: ipython2

    Image(filename='./images/image_files.png')




.. image:: images/output_3_0.png



.. code:: ipython2

    Image(filename='./defect_tensors/tp/9/tp_a9_c3/1.png')




.. image:: images/output_4_0.png



.. code:: ipython2

    testImage = cv2.imread('./defect_tensors/tp/9/tp_a9_c1/1.png')

.. code:: ipython2

    testImage




.. parsed-literal::

    array([[[104,  79,  79],
            [106,  81,  81],
            [104,  81,  81],
            ...,
            [  0,   0,   0],
            [  0,   0,   0],
            [  0,   0,   0]],

           [[105,  82,  82],
            [107,  84,  84],
            [108,  87,  86],
            ...,
            [  0,   0,   0],
            [  0,   0,   0],
            [  0,   0,   0]],

           [[107,  87,  86],
            [109,  88,  86],
            [111,  90,  88],
            ...,
            [ 92,  82,  79],
            [ 90,  81,  77],
            [ 95,  82,  80]],

           ...,
           [[100,  81,  87],
            [104,  83,  85],
            [106,  84,  85],
            ...,
            [ 98,  98, 103],
            [103, 100,  99],
            [  0,   0,   0]],

           [[  0,   0,   0],
            [  0,   0,   0],
            [  0,   0,   0],
            ...,
            [ 97,  98, 103],
            [101, 101, 100],
            [  0,   0,   0]],

           [[  0,   0,   0],
            [  0,   0,   0],
            [  0,   0,   0],
            ...,
            [ 99,  96,  99],
            [101,  95,  98],
            [  0,   0,   0]]], dtype=uint8)



.. code:: ipython2

    testImage.shape




.. parsed-literal::

    (105, 223, 3)



.. code:: ipython2

    plt.imshow(testImage)




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x108122d50>




.. image:: images/output_8_1.png


.. code:: ipython2

    plt.imshow(testImage[:,:,1])




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x106ad1a10>




.. image:: images/output_9_1.png


.. code:: ipython2

    testImage




.. parsed-literal::

    array([[[104,  79,  79],
            [106,  81,  81],
            [104,  81,  81],
            ...,
            [  0,   0,   0],
            [  0,   0,   0],
            [  0,   0,   0]],

           [[105,  82,  82],
            [107,  84,  84],
            [108,  87,  86],
            ...,
            [  0,   0,   0],
            [  0,   0,   0],
            [  0,   0,   0]],

           [[107,  87,  86],
            [109,  88,  86],
            [111,  90,  88],
            ...,
            [ 92,  82,  79],
            [ 90,  81,  77],
            [ 95,  82,  80]],

           ...,
           [[100,  81,  87],
            [104,  83,  85],
            [106,  84,  85],
            ...,
            [ 98,  98, 103],
            [103, 100,  99],
            [  0,   0,   0]],

           [[  0,   0,   0],
            [  0,   0,   0],
            [  0,   0,   0],
            ...,
            [ 97,  98, 103],
            [101, 101, 100],
            [  0,   0,   0]],

           [[  0,   0,   0],
            [  0,   0,   0],
            [  0,   0,   0],
            ...,
            [ 99,  96,  99],
            [101,  95,  98],
            [  0,   0,   0]]], dtype=uint8)



.. code:: ipython2

    plt.imshow(testImage[0:50, 50:100])




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x138c30790>




.. image:: images/output_11_1.png


**影像分類 Step by Step**

.. code:: ipython2

    Image(filename='./images/ETL.png')




.. image:: images/output_13_0.png



**Step1. 影像資料ETL**

.. code:: ipython2

    Image(filename='./images/initial_ETL.png')




.. image:: images/output_15_0.png



.. code:: ipython2

    tensor9, label9, sn9 = load_tensors('./defect_tensors/TP_Paper/9A/', width=256)


.. parsed-literal::

    (66, 154, 276)
    (31, 154, 276)
    (14, 154, 276)
    (14, 154, 276)


.. code:: ipython2

    plt.imshow(tensor9[2], cmap='gray')




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x138eccc90>




.. image:: images/output_17_1.png


.. code:: ipython2

    plt.imshow(tensor9[45], cmap='gray')




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x138c9c290>




.. image:: images/output_18_1.png


.. code:: ipython2

    tensor2, label2, sn2 = load_tensors('./defect_tensors/TP_Paper/2A/', width=256)


.. parsed-literal::

    (126, 242, 140)
    (17, 242, 140)
    (8, 242, 140)
    (2, 242, 140)


.. code:: ipython2

    tensor8, label8, sn8 = load_tensors('./defect_tensors/TP_Paper/8A/', width=256)


.. parsed-literal::

    (54, 297, 166)
    (51, 297, 166)
    (18, 297, 166)
    (18, 297, 166)


.. code:: ipython2

    tensor6, label6, sn6 = load_tensors('./defect_tensors/TP_Paper/6A/', width=256)


.. parsed-literal::

    (125, 157, 267)
    (4, 157, 267)
    (9, 157, 267)
    (1, 157, 267)


**確認載入的圖片是否有問題?**

.. code:: ipython2

    from defectlib import display_tensor

.. code:: ipython2

    display_tensor(tensor9, label9, sn9)



.. image:: images/output_24_0.png


**合併所有角度的照片**

.. code:: ipython2

    from defectlib import combine_shuffle_tensors

.. code:: ipython2

    for tensor in (tensor2, tensor6, tensor8, tensor9):
        print tensor.shape


.. parsed-literal::

    (153, 256, 256)
    (139, 256, 256)
    (141, 256, 256)
    (125, 256, 256)


.. code:: ipython2

    Image(filename='./images/combine_shuffle_tensors.png')




.. image:: images/output_28_0.png



.. code:: ipython2

    all_tensor, all_label, all_sn = combine_shuffle_tensors((tensor2, label2, sn2),
                                                           (tensor6, label6, sn6),
                                                           (tensor8, label8, sn8),
                                                           (tensor9, label9, sn9))


.. parsed-literal::

    the final tensor should be 558
    number of class 0: 371
    	number of SN: 15
    number of class 1: 103
    	number of SN: 3
    number of class 2: 49
    	number of SN: 1
    number of class 3: 35
    	number of SN: 1


.. code:: ipython2

    for label in set(all_label):
        print 'the number of class {}: {}'.format(label, all_tensor[all_label == label].shape[0])


.. parsed-literal::

    the number of class 0: 371
    the number of class 1: 103
    the number of class 2: 49
    the number of class 3: 35


.. code:: ipython2

    all_tensor.shape




.. parsed-literal::

    (558, 256, 256)



.. code:: ipython2

    all_label.shape




.. parsed-literal::

    (558,)



.. code:: ipython2

    display_tensor(all_tensor, all_label, all_sn)



.. image:: images/output_33_0.png


**開始建立模型**

.. code:: ipython2

    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Flatten, Activation
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.utils import np_utils

.. code:: ipython2

    list(set(all_label))




.. parsed-literal::

    [0, 1, 2, 3]



.. code:: ipython2

    labels_ohe = np_utils.to_categorical(all_label)

.. code:: ipython2

    labels_ohe




.. parsed-literal::

    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  1.,  0.,  0.],
           ...,
           [ 1.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.]])



.. code:: ipython2

    all_tensor.shape




.. parsed-literal::

    (558, 256, 256)



.. code:: ipython2

    Image(filename='./images/CNN.png')




.. image:: images/output_40_0.png



.. code:: ipython2

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
    model.add(Dense(4))
    model.add(Activation('softmax'))

.. code:: ipython2

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])

.. code:: ipython2

    combined_tensor8, combined_label8, combined_sn8 = combine_shuffle_tensors((tensor2, label2, sn2),
                                                                             (tensor6, label6, sn6),
                                                                             (tensor9, label9, sn9))


.. parsed-literal::

    the final tensor should be 417
    number of class 0: 317
    	number of SN: 15
    number of class 1: 52
    	number of SN: 3
    number of class 2: 31
    	number of SN: 1
    number of class 3: 17
    	number of SN: 1


.. code:: ipython2

    train_data = combined_tensor8.reshape(combined_tensor8.shape[0], combined_tensor8.shape[1], combined_tensor8.shape[2], 1)

.. code:: ipython2

    train_data.shape




.. parsed-literal::

    (417, 256, 256, 1)



.. code:: ipython2

    train_labels = np_utils.to_categorical(combined_label8)

.. code:: ipython2

    test_data = tensor8.reshape(tensor8.shape[0], tensor8.shape[1], tensor8.shape[2], 1)

.. code:: ipython2

    test_labels = np_utils.to_categorical(label8)

.. code:: ipython2

    %time
    nb_epoch = 10  # try increasing this number
    model.fit(train_data, train_labels, batch_size=10, nb_epoch=nb_epoch,
              show_accuracy=True, verbose=1, validation_data=(test_data, test_labels))
    score = model.evaluate(test_data, test_labels, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


.. parsed-literal::

    CPU times: user 3 µs, sys: 3 µs, total: 6 µs
    Wall time: 8.82 µs
    Train on 417 samples, validate on 141 samples
    Epoch 1/10
    417/417 [==============================] - 69s - loss: 0.6262 - acc: 0.8129 - val_loss: 0.8284 - val_acc: 0.7447
    Epoch 2/10
    417/417 [==============================] - 66s - loss: 0.3379 - acc: 0.9137 - val_loss: 0.5481 - val_acc: 0.8156
    Epoch 3/10
    417/417 [==============================] - 66s - loss: 0.1746 - acc: 0.9329 - val_loss: 0.7005 - val_acc: 0.8156
    Epoch 4/10
    417/417 [==============================] - 66s - loss: 0.0794 - acc: 0.9712 - val_loss: 0.6886 - val_acc: 0.8227
    Epoch 5/10
    417/417 [==============================] - 66s - loss: 0.0676 - acc: 0.9736 - val_loss: 0.8805 - val_acc: 0.8156
    Epoch 6/10
    417/417 [==============================] - 66s - loss: 0.0402 - acc: 0.9904 - val_loss: 0.8275 - val_acc: 0.8440
    Epoch 7/10
    417/417 [==============================] - 66s - loss: 0.0432 - acc: 0.9880 - val_loss: 0.7888 - val_acc: 0.8369
    Epoch 8/10
    417/417 [==============================] - 66s - loss: 0.0313 - acc: 0.9928 - val_loss: 0.9393 - val_acc: 0.8440
    Epoch 9/10
    417/417 [==============================] - 66s - loss: 0.0172 - acc: 0.9952 - val_loss: 0.9032 - val_acc: 0.8440
    Epoch 10/10
    417/417 [==============================] - 66s - loss: 0.0167 - acc: 0.9976 - val_loss: 1.2729 - val_acc: 0.8369


.. parsed-literal::

    /Users/hadoop1/.virtualenvs/cv/lib/python2.7/site-packages/keras/models.py:651: UserWarning: The "show_accuracy" argument is deprecated, instead you should pass the "accuracy" metric to the model at compile time:
    `model.compile(optimizer, loss, metrics=["accuracy"])`
      warnings.warn('The "show_accuracy" argument is deprecated, '


.. parsed-literal::

    ('Test score:', 1.2729044203904081)
    ('Test accuracy:', 0.83687943262411346)

