import numpy as np
import os, pickle, yaml, tarfile
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from dvclive import Live

OUTPUT_DIR = "output"
fpath = os.path.join(OUTPUT_DIR, "data.pkl")
with open(fpath, "rb") as fd:
    data = pickle.load(fd)
(x_train, y_train),(x_test, y_test) = data

unique, counts = np.unique(y_train, return_counts=True)
print("Train labels: ", dict(zip(unique, counts)))
unique, counts = np.unique(y_test, return_counts=True)
print("\nTest labels: ", dict(zip(unique, counts)))

num_labels = len(np.unique(y_train))
y_train = to_categorical(y_train)

image_size = x_train.shape[1]
input_size = image_size * image_size

params = yaml.safe_load(open("params.yaml"))["train"]
batch_size = params["batch_size"]
hidden_units = params["hidden_units"]
dropout = params["dropout"]
num_epochs = params["num_epochs"]
lr = params["lr"]
conv_activation = params["conv_activation"]

x_train = x_train.reshape(-1, 28, 28, 1)
x_train = x_train.astype('float32') / 255
model = Sequential()
model.add(Conv2D(filters=28, kernel_size=(3,3), activation=conv_activation))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(hidden_units, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(num_labels, activation='softmax'))

opt = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(loss='categorical_crossentropy', 
              optimizer=opt,
              metrics=['accuracy'])

class AccuracyHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.per_epoch_accuracies = []

    def on_epoch_end(self, epoch, logs):
        accuracy = logs.get("accuracy")
        self.per_epoch_accuracies.append(accuracy)
        print("\n\nPer epoch accuracies = \n\n", self.per_epoch_accuracies)
        live.log_metric("accuracy", accuracy)
        live.next_step()

with Live() as live:
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, callbacks=[AccuracyHistory()], verbose=1)

    model.summary()

    model_keras_path = 'myfmmodel.keras'
    model_path = os.path.join(OUTPUT_DIR, model_keras_path)
    model.save(model_path)
    # live.log_artifact(model_path, type="model", cache=False)

# Save model archive
model_version = '1' # Since we use Studio/GTO for model versioning, I'm hard-coding this to 1
export_dir = 'export/Servo/' + model_version
tf.saved_model.save(model, export_dir)
model_archive_path = 'myfmmodel.tar.gz'
model_archive = f'{OUTPUT_DIR}/{model_archive_path}'
with tarfile.open(model_archive, mode='w:gz') as archive:
    archive.add('export', recursive=True)

def history_to_csv(history):
    # This code is copied from https://github.com/iterative/get-started-experiments/
    keys = list(history.history.keys())
    csv_string = ",".join(["epoch"] + keys) + "\n"
    list_len = len(history.history[keys[0]])
    for i in range(list_len):
        row = (
            str(i + 1)
            + ","
            + ",".join([str(history.history[k][i]) for k in keys])
            + "\n"
        )
        csv_string += row

    return csv_string
print(history)
with open("output/train_logs.csv", "w") as f:
    f.write(history_to_csv(history))
