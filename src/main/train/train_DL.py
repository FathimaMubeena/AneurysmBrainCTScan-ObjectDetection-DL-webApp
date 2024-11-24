import datetime
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from datetime import datetime
import os


def main():
    parent_dir = os.path.dirname(os.getcwd())
    data_directory = os.path.join(parent_dir, 'resources/kaggleData/files/')

    # Image size, batch, channels
    IMAGE_SIZE = 256
    BATCH = 32
    CHANNELS = 3

    # Load dataset using tf.keras.preprocessing.image_dataset_from_directory
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_directory,
        shuffle=True,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH
    )

    label_classes = dataset.class_names
    print(label_classes)


    # Data augmentation : Define the image preprocessing pipeline with data augmentation
    image_preprocessing = tf.keras.Sequential([
        # Resize the images to a fixed size
        layers.Resizing(IMAGE_SIZE, IMAGE_SIZE, interpolation='bilinear', crop_to_aspect_ratio=False),

        # Rescale pixel values to [0, 1] range
        layers.Rescaling(1. / 255),

        # Apply data augmentation techniques
        layers.RandomFlip("horizontal_and_vertical"),  # Randomly flip images horizontally and vertically
        layers.RandomRotation(0.2),                    # Randomly rotate images by up to 20%
        layers.RandomZoom(0.1),                        # Optional: Randomly zoom images by up to 10%
        layers.RandomContrast(0.2)                     # Optional: Adjust image contrast
    ])

    # Splitting the dataset into train, test, and validation
    def split_train_test_val(ds, train_split=0.8, test_split=0.1, val_split=0.1):
        ds_size = len(ds)
        train_size = round(train_split * ds_size)
        test_size = round(test_split * ds_size)
        val_size = round(val_split * ds_size)

        train_ds = ds.take(train_size)
        test_ds = ds.skip(train_size).take(test_size)
        val_ds = ds.skip(train_size).skip(test_size)
        return train_ds, test_ds, val_ds

    # calling split_train_test_val() function
    train_ds, test_ds, val_ds = split_train_test_val(dataset)

    # CNN Model definition
    model = models.Sequential([
        # Specify the input shape for the first layer (Conv2D layer)
        layers.InputLayer(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),  # Define input shape explicitly
        image_preprocessing, # Image Preprocessing layer
        layers.Conv2D(32, (8, 8), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (4, 4), activation='relu'),
        layers.Dropout(0.2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(len(label_classes), activation='softmax') # Output layer
    ])

    # Compile the model
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    # Train the model
    EPOCHS = 50
    history = model.fit(train_ds,
                        epochs=EPOCHS,
                        validation_data=val_ds)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    print(f'Test Loss: {test_loss * 100:.2f}%')

    export_deploy_dir = os.path.join(parent_dir, 'deploy/')
    # Get the current date
    current_date = datetime.now().strftime("%Y%m%d")

    # Save the model with the date appended to the filename
    model.save(os.path.join(export_deploy_dir, f'model_{current_date}.keras'))


if __name__ == "__main__":
    main()