# Import library
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
import pandas as pd

def main() -> None:

    train_datagen = ImageDataGenerator(rescale=1/255.)
    val_datagen = ImageDataGenerator(rescale=1/255.)

    train_data = train_datagen.flow_from_directory(directory="files/train",
                                                target_size=(240, 240),
                                                batch_size=32,
                                                class_mode="categorical")

    validation_data = val_datagen.flow_from_directory(directory="files/validation",
                                                target_size=(240, 240),
                                                batch_size=32,
                                                class_mode="categorical")


    test_datagen = ImageDataGenerator(rescale=1/255.)
    test_data = test_datagen.flow_from_directory(directory="GTSRB/test",
                                                target_size=(240, 240),
                                                batch_size=32,
                                                class_mode="categorical")

    """## Create and fit the model"""


    # Set random seed
    tf.random.set_seed(42)



    # Create model
    model = Sequential([
        Conv2D(16, 3, activation="relu", input_shape=(240, 240, 3)),
        MaxPool2D(pool_size=2),
        Conv2D(16, 3, activation="relu"),
        MaxPool2D(pool_size=2),
        Flatten(),
        Dense(43, activation="softmax")
    ])




    # Compile the model
    model.compile(loss="categorical_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

    model.summary()

    # Fit the model

    batch_sizes = 64
    epoch = 3
    trainsteps = (train_data.n//batch_sizes)
    valsteps = (validation_data.n//batch_sizes)
    print(train_data.n)
    print(validation_data.n)
    print(trainsteps)
    print(valsteps)
    print(train_data)

    history = model.fit(train_data,
                        batch_size=batch_sizes,
                        epochs=epoch,
                        steps_per_epoch=trainsteps,
                        validation_data=validation_data,
                        validation_steps=valsteps)


    # Evaluate model

#model.evaluate(test_data)


    pd.DataFrame(history.history).plot()

    model.save('og_model.keras')



if __name__ == '__main__':
    main()
