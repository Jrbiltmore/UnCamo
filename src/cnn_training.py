# Script for CNN training for epithelial modulation detection
# CNN Training Module for Evolved Epithelial Modulation Detection

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class CNNTrainer:
    def __init__(self, input_shape=(128, 128, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        '''
        Build a Convolutional Neural Network (CNN) model.
        
        Returns:
            tf.keras.Model: Compiled CNN model.
        '''
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_dir, val_dir, batch_size=32, epochs=50, checkpoint_path='best_model.h5'):
        '''
        Train the CNN model using image data generators.
        
        Args:
            train_dir (str): Directory containing training images.
            val_dir (str): Directory containing validation images.
            batch_size (int): Size of image batches for training.
            epochs (int): Number of training epochs.
            checkpoint_path (str): Filepath to save the best model.
        
        Returns:
            tf.keras.callbacks.History: Training history object.
        '''
        train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15,
                                           width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                                           horizontal_flip=True, fill_mode="nearest")

        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(train_dir, target_size=self.input_shape[:2],
                                                            batch_size=batch_size, class_mode='categorical')

        val_generator = val_datagen.flow_from_directory(val_dir, target_size=self.input_shape[:2],
                                                        batch_size=batch_size, class_mode='categorical')

        # Callbacks for early stopping and saving the best model
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')
        ]

        history = self.model.fit(train_generator, epochs=epochs, validation_data=val_generator,
                                 callbacks=callbacks)

        return history

    def evaluate(self, test_dir, batch_size=32):
        '''
        Evaluate the trained CNN model on a test dataset.
        
        Args:
            test_dir (str): Directory containing test images.
            batch_size (int): Size of image batches for evaluation.
        
        Returns:
            dict: Evaluation results containing loss and accuracy.
        '''
        test_datagen = ImageDataGenerator(rescale=1./255)

        test_generator = test_datagen.flow_from_directory(test_dir, target_size=self.input_shape[:2],
                                                          batch_size=batch_size, class_mode='categorical')

        results = self.model.evaluate(test_generator)
        return dict(zip(self.model.metrics_names, results))

    def predict(self, image_path):
        '''
        Predict the class of a single image using the trained CNN model.
        
        Args:
            image_path (str): Filepath of the image to be predicted.
        
        Returns:
            np.array: Predicted class probabilities.
        '''
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.input_shape[:2])
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

        predictions = self.model.predict(img_array)
        return predictions

    def visualize_training(self, history):
        '''
        Visualize the training accuracy and loss over epochs.
        
        Args:
            history (tf.keras.callbacks.History): Training history object.
        
        Returns:
            None: Displays the training and validation loss/accuracy plots.
        '''
        import matplotlib.pyplot as plt

        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.show()

    def save_model(self, filepath='cnn_model.h5'):
        '''
        Save the trained CNN model to a file.
        
        Args:
            filepath (str): Filepath to save the model.
        
        Returns:
            None
        '''
        self.model.save(filepath)
    
    def load_model(self, filepath='cnn_model.h5'):
        '''
        Load a pre-trained CNN model from a file.
        
        Args:
            filepath (str): Filepath from which to load the model.
        
        Returns:
            None
        '''
        self.model = tf.keras.models.load_model(filepath)

    def fine_tune_model(self, base_model_path, train_dir, val_dir, batch_size=32, epochs=10):
        '''
        Fine-tune a pre-trained CNN model by further training on new data.
        
        Args:
            base_model_path (str): Path to the pre-trained model.
            train_dir (str): Directory containing training images.
            val_dir (str): Directory containing validation images.
            batch_size (int): Size of image batches for training.
            epochs (int): Number of fine-tuning epochs.
        
        Returns:
            tf.keras.callbacks.History: Training history object.
        '''
        # Load the base model
        self.load_model(base_model_path)

        # Unfreeze the last few layers for fine-tuning
        for layer in self.model.layers[-4:]:
            layer.trainable = True

        # Compile the model again with a lower learning rate for fine-tuning
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

        # Use data generators for new data
        train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15,
                                           width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                                           horizontal_flip=True, fill_mode="nearest")

        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(train_dir, target_size=self.input_shape[:2],
                                                            batch_size=batch_size, class_mode='categorical')

        val_generator = val_datagen.flow_from_directory(val_dir, target_size=self.input_shape[:2],
                                                        batch_size=batch_size, class_mode='categorical')

        # Fine-tune the model
        history = self.model.fit(train_generator, epochs=epochs, validation_data=val_generator)

        return history
