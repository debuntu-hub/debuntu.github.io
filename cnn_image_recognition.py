import argparse
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical


def build_model(input_shape):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main(epochs: int):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # reshape for channels last
    x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = build_model((28, 28, 1))
    model.fit(x_train, y_train, epochs=epochs, validation_split=0.1)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple CNN for MNIST image recognition')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train the model')
    args = parser.parse_args()
    main(args.epochs)
