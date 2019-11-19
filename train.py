import argparse
import json
import numpy as np
import os
import tensorflow.keras as keras


def load_data(input_path):
    path = os.path.join(input_path, 'mnist/mnist.npz')
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return {
        'x_train': x_train,
        'y_train': y_train, 
        'x_test': x_test,
        'y_test': y_test,
    }


def create_model(params):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax'), # why ,?
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train(model, data, params):
    def print_metrics(epoch, logs):
        print()
        logs = {k: str(v) for (k, v) in logs.items()}
        print(json.dumps({'epoch': epoch, **logs}))

    print_metrics_callback = keras.callbacks.LambdaCallback(on_epoch_end=print_metrics)

    model.fit(data['x_train'], data['y_train'], epochs=params.epochs, callbacks=[print_metrics_callback])


def evaluate(model, data):
    test_loss, test_acc = model.evaluate(data['x_test'], data['y_test'])
    print(json.dumps({
        'test_loss': str(test_loss),
        'test_acc': str(test_acc),
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)    
    parser.add_argument('--learning_rate', type=float, default=0.001)
    params = parser.parse_args()

    input_path = os.getenv('VH_INPUTS_DIR', '/valohai/inputs')
    output_path = os.getenv('VH_OUTPUTS_DIR', '/valohai/outputs')

    data = load_data(input_path)
    model = create_model(params)
    train(model, data, params)
    model.save(os.path.join(output_path, 'mnist.h5'))
    evaluate(model, data)
