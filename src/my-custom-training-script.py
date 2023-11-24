import logging
import json
import os
import glob
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
from random import sample

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    logger.debug('Hello my custom SageMaker init script!')
    f_output_model = open("/opt/ml/model/my-model.txt", "w")
    f_output_model.write(json.dumps(glob.glob("{}/*/*/*.*".format(os.environ['SM_INPUT_DIR']))))
    f_output_model.close()
    
    f_output_data = open("/opt/ml/output/data/my-data.txt", "w")
    f_output_data.write(json.dumps(dict(os.environ), sort_keys=True, indent=4))
    f_output_data.close()

    # Caricamento e preprocessing delle immagini
    def load_images_from_folder(folder_path):
        images = []
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            images.append(img)
        return images

    base_path = "{}/data/training/deeplearningdataset_2023-10-17".format(os.environ["SM_INPUT_DIR"])
    closed_train_folder = os.path.join(base_path, 'TrainingSet', 'TrainingSet', 'Closed')
    opened_train_folder = os.path.join(base_path, 'TrainingSet', 'TrainingSet', 'Opened')
    closed_test_folder = os.path.join(base_path, 'TestSet', 'TestSet', 'Closed')
    opened_test_folder = os.path.join(base_path, 'TestSet', 'TestSet', 'Opened')

    # Caricamento delle immagini
    closed_train_images = load_images_from_folder(closed_train_folder)
    opened_train_images = load_images_from_folder(opened_train_folder)
    closed_test_images = load_images_from_folder(closed_test_folder)
    opened_test_images = load_images_from_folder(opened_test_folder)

    # Creazione delle etichette
    X_train = np.array(closed_train_images + opened_train_images)
    y_train = np.array([0] * len(closed_train_images) + [1] * len(opened_train_images))
    X_test = np.array(closed_test_images + opened_test_images)
    y_test = np.array([0] * len(closed_test_images) + [1] * len(opened_test_images))

   


    # Dividiamo ulteriormente il training set in un "vero" training set e un validation set.
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=25,  
        width_shift_range=0.25,  
        height_shift_range=0.25,  
        horizontal_flip=True,
        brightness_range=[0.4, 1.6],  
        zoom_range=0.35,  
        fill_mode='nearest',
        channel_shift_range=0.55  
    )

    def create_model():
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=(128, 128, 3)))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        model.add(MaxPooling2D(2, 2))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        model.add(MaxPooling2D(2, 2))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        model.add(MaxPooling2D(2, 2))

        model.add(Flatten())

        model.add(Dense(1024, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.ReLU())

        model.add(Dense(1, activation='sigmoid'))

        optimizer = Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.00001, verbose=1)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    model = create_model()
    train_data = datagen.flow(X_train, y_train, batch_size=32)
    val_data = (X_val, y_val)

    history = model.fit(train_data, validation_data=val_data, epochs=50, callbacks=[early_stop, reduce_lr, model_checkpoint])

    # Evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Salviamo
    model.save('eye_state_model2.h5')


    # Carico miglior modello
    best_model = load_model('best_model2.h5')
    test_loss, test_acc = best_model.evaluate(X_test, y_test)
    print(f"Test Accuracy with Best Model: {test_acc * 100:.2f}%")

    # F1-score e Confusion Matrix
    y_pred = (best_model.predict(X_test) > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred)

    print(f"F1 Score: {f1:.4f}")


    # addestramento intensivo sul miglior modello
    best_model.fit(train_data, validation_data=val_data, epochs=50, callbacks=[early_stop, reduce_lr, model_checkpoint])

    # Salviamo il modello finale addestrato
    model.save('ultimate_model2.h5')

