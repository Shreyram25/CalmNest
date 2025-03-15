import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define emotion labels (FER2013 dataset labels)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def download_dataset():
    """Download FER2013 dataset if not already present"""
    if not os.path.exists('fer2013.csv'):
        print("Downloading FER2013 dataset...")
        url = "https://www.dropbox.com/s/4543c4evfwm8gc7/fer2013.csv?dl=1"
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open('fer2013.csv', 'wb') as file, tqdm(
            desc='Downloading',
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)

def load_fer2013():
    """Load and preprocess FER2013 dataset"""
    # Download dataset if needed
    download_dataset()
    
    # Read the CSV file
    data = pd.read_csv('fer2013.csv')
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    
    for pixel_sequence in tqdm(pixels, desc='Processing images'):
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = face.astype('float32')
        faces.append(face)

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    
    emotions = pd.get_dummies(data['emotion']).values
    
    return faces, emotions

def create_model():
    """Create and compile the emotion detection model"""
    model = Sequential([
        Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 emotion classes
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def train_model():
    """Train the emotion detection model"""
    # Create model directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Load and preprocess data
    print("Loading dataset...")
    faces, emotions = load_fer2013()
    
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.2, random_state=42)
    
    # Data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Create and compile model
    model = create_model()
    
    # Define callbacks
    checkpoint = ModelCheckpoint('models/emotion_model_best.h5',
                               monitor='val_accuracy',
                               save_best_only=True,
                               mode='max',
                               verbose=1)
    
    early_stopping = EarlyStopping(monitor='val_loss',
                                 patience=10,
                                 restore_best_weights=True)
    
    # Train the model
    print("Training model...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        epochs=50,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint, early_stopping]
    )
    
    # Save the final model
    model.save('models/emotion_model_final.h5')
    print("Training completed! Model saved as 'models/emotion_model_final.h5'")
    
    # Evaluate the model
    print("\nEvaluating model on test set:")
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

if __name__ == "__main__":
    train_model() 