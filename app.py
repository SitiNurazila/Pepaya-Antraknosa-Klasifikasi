from flask import Flask, render_template, jsonify, request
import os, shutil, random
from shutil import copyfile

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('agg')
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2 as cv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc, roc_auc_score
import cv2
from keras.models import load_model


app = Flask(__name__)
app.secret_key = "cnn_klasifikasi"
model = ""

@app.route('/', methods=['GET', 'POST'])
def index():
    train_antraknosa, train_nonantraknosa, val_antraknosa, val_nonantraknosa, antraknosa, nonantraknosa = data()

    return render_template('index.html', antraknosa = antraknosa, nonantraknosa = nonantraknosa, 
                           train_antraknosa = train_antraknosa, train_nonantraknosa=train_nonantraknosa, 
                           val_antraknosa = val_antraknosa,val_nonantraknosa = val_nonantraknosa)

@app.route('/cnn')
def indexku():
    return render_template('cnn.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        upload_folder = "static/testing"
        app.config['UPLOAD_FOLDER'] = upload_folder
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        path = "static/testing/" + f.filename
        hasil = predict(path)
        return render_template('cnn.html', hasil = hasil, gambar = path)
    
@app.route('/prosescnn', methods=['GET', 'POST'])
def training():
    train_antraknosa, train_nonantraknosa, val_antraknosa, val_nonantraknosa, antraknosa, nonantraknosa = data()
    if request.method == 'POST':
        epochs = int(request.form['epoch'])
        optm = request.form['opt']
        if optm == 'sgd':
            opt = tf.optimizers.SGD()
        elif optm == 'adam':
            opt = tf.optimizers.Adam()
        elif optm == 'nadam':
            opt = tf.optimizers.Nadam()
        train_generator, validation_generator = prep()
        history, acc_train, acc_test, loss_train, loss_test= model(train_generator, validation_generator,epochs, opt, optm)
        path_acc, path_loss = ploting(history, optm, epochs)
    return render_template('index.html', acc_train = acc_train, acc_test = acc_test, loss_train = loss_train,
                        loss_test = loss_test, akurasi = path_acc, loss = path_loss, antraknosa = antraknosa, 
                        nonantraknosa = nonantraknosa, train_antraknosa = train_antraknosa, 
                        train_nonantraknosa=train_nonantraknosa, val_antraknosa = val_antraknosa,
                        val_nonantraknosa = val_nonantraknosa)
def predict(fn):
    # modelcnn = load_model('E:/flask/PP/cnn_fix_20.h5')
    modelcnn = load_model('static/images/grafik/ADAM/EPOCH 20/P1/adam20cnn.h5')
    path = fn 
    img = image.load_img(path, target_size = (200,150))
    x = image.img_to_array(img)
    x= np.expand_dims(x, axis = 0)

    images = np.vstack([x])
    classes = modelcnn.predict(images, batch_size = 10)
    print(fn)
    print(classes[0])
    if classes[0]<0.5:
        hasil = "antraknosa"
    else:
        hasil ="nonantraknosa"
    return f'ini termasuk dalam kelas {hasil}'
def data ():
    train_antraknosa_dir = os.path.join('E:/flask/ICP/Dataset/train/antraknosa')
    train_nonantraknosa_dir = os.path.join('E:/flask/ICP/Dataset/train/nonantraknosa')
    valid_antraknosa_dir = os.path.join('E:/flask/ICP/Dataset/validasi/antraknosa')
    valid_nonantraknosa_dir = os.path.join('E:/flask/ICP/Dataset/validasi/nonantraknosa')
    train_antraknosa = len(os.listdir(train_antraknosa_dir))
    train_nonantraknosa = len(os.listdir(train_nonantraknosa_dir))
    val_antraknosa = len(os.listdir(valid_antraknosa_dir))
    val_nonantraknosa = len(os.listdir(valid_nonantraknosa_dir))
    antraknosa = train_antraknosa+ val_antraknosa
    nonantraknosa = train_nonantraknosa+ val_nonantraknosa
    return train_antraknosa, train_nonantraknosa, val_antraknosa, val_nonantraknosa, antraknosa, nonantraknosa
def prep():
    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1/255)
    validation_datagen = ImageDataGenerator(rescale=1/255)

    # Flow training images in batches of 120 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
            'E:/flask/ICP/Dataset/train',  # This is the source directory for training images
            classes = ['antraknosa', 'nonantraknosa'],
            target_size=(200, 150),  # All images will be resized to 200x150
            batch_size=30,
            # Use binary labels
            class_mode='binary')

    # Flow validation images in batches of 19 using valid_datagen generator
    validation_generator = validation_datagen.flow_from_directory(
            'E:/flask/ICP/Dataset/train',  # This is the source directory for training images
            classes = ['antraknosa', 'nonantraknosa'],
            target_size=(200, 150),  # All images will be resized to 200x150
            batch_size=30,
            # Use binary labels
            class_mode='binary',
            shuffle=False)
    return train_generator, validation_generator
def model(train_generator, validation_generator,epochs, opt, optm):
    act = 'sigmoid'
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu',
                            input_shape=(200, 150, 3),
                            bias_initializer='zeros',
                            kernel_initializer='he_normal',
                            padding = 'same'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(32, (3,3), activation='relu',
                            input_shape=(200, 150, 3),
                            bias_initializer='zeros',
                            kernel_initializer='he_normal',
                            padding = 'same'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu',
                            input_shape=(150, 150, 3),
                            bias_initializer='zeros',
                            kernel_initializer='he_normal',
                            padding = 'same'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu',
                            input_shape=(150, 150, 3),
                            bias_initializer='zeros',
                            kernel_initializer='he_normal',
                            padding = 'same'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation=act)
    ])
    model.summary()
    # tf.keras.utils.plot_model(model)

    model.compile(optimizer = opt,
                loss = 'binary_crossentropy',
                metrics=['accuracy'])
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end (self, epoch, logs = {}):
            if(logs.get('accuracy') == 1):
                    self.model.stop_training = True
                    print('\nAkurasi mencapai 100%')
    callbacks = myCallback()

    history = model.fit(train_generator,
        epochs=epochs,
        verbose=1,
        validation_data = validation_generator,
        callbacks = [callbacks]
    )
    val = model.evaluate(validation_generator)
    acc_train = history.history['accuracy']
    acc_train = acc_train[-1]
    acc_train = "{:.2f}%".format(acc_train * 100)
    acc_test = val[1]
    acc_test = "{:.2f}%".format(acc_test * 100)
    loss_train = history.history['loss']
    loss_train = loss_train[-1]
    loss_train = "{:.4f}".format(loss_train)
    loss_test = val[0]
    loss_test = "{:.4f}".format(loss_test)
    save_dir=r'/ICP'
    save_id=str ('cnn.h5')
    save_loc=os.path.join(save_dir, save_id)
    model.save(save_loc)
    epochstr = str(epochs)
    model.save(optm + epochstr +'cnn.h5')
    return history, acc_train, acc_test, loss_train, loss_test

def ploting(history, optm, epochs):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    epochstr = str(epochs)
    path_acc = "E:/flask/PP1/static/images/grafik/" + optm + epochstr + "accuracy.png"
    plt.savefig(path_acc)
    plt.close()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    path_loss = "E:/flask/PP1/static/images/grafik/" + optm + epochstr +"loss.png"
    plt.savefig(path_loss)
    plt.close()
    return path_acc, path_loss



app.run(host='127.0.0.1', port='80', debug=True)