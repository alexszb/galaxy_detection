from email.mime import base
from gc import callbacks
import shutil
import tensorflow as tf
from keras import models, layers
from keras_preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from keras.callbacks import CSVLogger
import numpy as np
import os
import pathlib
from PIL import Image
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, f1_score
import matplotlib.pyplot as plt

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []
        self.auc = []
        self.precision = []
        self.recall = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.auc.append(logs.get('auc'))
        self.precision.append(logs.get("precision"))
        self.recall.append(logs.get("recall"))

project_dir = os.path.dirname(__file__)
data_dir = pathlib.Path("%s/data/" % project_dir)
solution_dir = pathlib.Path(project_dir).parent

# ringed_spiral_files = os.listdir("%s/images/ringed_spirals/clean" % solution_dir)
# non_ringed_spiral_files = os.listdir("%s/images/non_ringed_spirals/clean" % solution_dir)

# train_spiral_files = ringed_spiral_files[0:int(len(ringed_spiral_files)*0.7)]
# val_spiral_files = ringed_spiral_files[int(len(ringed_spiral_files)*0.7):int(len(ringed_spiral_files)*0.9)]
# test_spiral_files = ringed_spiral_files[int(len(ringed_spiral_files)*0.9):int(len(ringed_spiral_files))]

# train_non_ringed_spiral_files = non_ringed_spiral_files[0:int(len(non_ringed_spiral_files)*0.7)]
# val_non_ringed_spiral_files = non_ringed_spiral_files[int(len(non_ringed_spiral_files)*0.7):int(len(non_ringed_spiral_files)*0.9)]
# test_non_ringed_spiral_files = non_ringed_spiral_files[int(len(non_ringed_spiral_files)*0.9):int(len(non_ringed_spiral_files))]

# for f in train_spiral_files:
#   shutil.copy("%s/images/ringed_spirals/clean/%s" % (solution_dir, f), "%s/train/ringed_spiral/%s" % (data_dir, f))
# for f in val_spiral_files:
#   shutil.copy("%s/images/ringed_spirals/clean/%s" % (solution_dir, f), "%s/val/ringed_spiral/%s" % (data_dir, f))
# for f in test_spiral_files:
#   shutil.copy("%s/images/ringed_spirals/clean/%s" % (solution_dir, f), "%s/test/ringed_spiral/%s" % (data_dir, f))

# for f in train_non_ringed_spiral_files:
#   shutil.copy("%s/images/non_ringed_spirals/clean/%s" % (solution_dir, f), "%s/train/non_ringed_spiral/%s" % (data_dir, f))
# for f in val_non_ringed_spiral_files:
#   shutil.copy("%s/images/non_ringed_spirals/clean/%s" % (solution_dir, f), "%s/val/non_ringed_spiral/%s" % (data_dir, f))
# for f in test_non_ringed_spiral_files:
#   shutil.copy("%s/images/non_ringed_spirals/clean/%s" % (solution_dir, f), "%s/test/non_ringed_spiral/%s" % (data_dir, f))

batch_size = 32
img_height = 128
img_width = 128

train_dir = pathlib.Path("%s/data/train/" % project_dir)
train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_dir = pathlib.Path("%s/data/val/" % project_dir)
val_ds = tf.keras.utils.image_dataset_from_directory(
  val_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_dir = pathlib.Path("%s/data/test/" % project_dir)
# train_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   subset="training",
#   validation_split=0.2,
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

# val_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="validation",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

class_names = train_ds.class_names
num_classes = len(class_names)
print(class_names)

# AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.cache().shuffle(200).prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential(
  [
    # layers.Cropping2D(cropping=((80,80), (80,80)), input_shape=(318,318,3)),
    layers.RandomFlip("horizontal_and_vertical",
                      input_shape=(128,
                                  128,
                                  3)),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(height_factor=(-0.1,0.1),width_factor=(-0.1,0.1), fill_mode="nearest"),
  ]
)

datagen = ImageDataGenerator(
        # rotation_range=45,
        width_shift_range=0.1,
        height_shift_range=0.1,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True,
        # vertical_flip=True,
        # brightness_range=(0.7, 1.0),
        fill_mode='nearest')

# train_generator = datagen.flow_from_directory(
#         train_dir,  # this is the target directory
#         target_size=(128, 128),  # all images will be resized to 150x150
#         batch_size=batch_size,
#         class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# validation_generator = datagen.flow_from_directory(
#         val_dir,
#         target_size=(128, 128),
#         batch_size=batch_size,
#         class_mode='binary')

test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='binary')

base_model = tf.keras.applications.ResNet50(weights = 'imagenet', include_top = False, input_shape = (128,128,3))
# for layer in base_model.layers:
#   layer.trainable = False

inp = layers.Input((128,128,3))
x = data_augmentation(inp)
x = base_model(x)
# x = layers.Flatten()(x)
# x = layers.Dense(512, activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)
predictions = layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs = inp, outputs = predictions)

train_checkpointer = tf.keras.callbacks.ModelCheckpoint(monitor="val_precision", save_best_only=True, mode="max", filepath="%s/cnn_ring_chk.hdf5"% project_dir)
train_csv_logger = CSVLogger('train_log_epoch.csv', append=False, separator=';')
def lr_schedule(epoch):
    if epoch < 5:
      return 0.01
    else:
      return 0.01*(1-0.9*(epoch-4)/19)

model.compile(optimizer='SGD',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits = False),
              metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# model.summary()
# plot_model(model, to_file='%s/CNN_ring_plot.png' % project_dir, show_shapes=True, show_layer_names=True, show_layer_activations=True)

train_history = LossHistory()

epochs=20
# history = model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs,
#   callbacks=[train_checkpointer, train_csv_logger, train_history,tf.keras.callbacks.LearningRateScheduler(lr_schedule)]
# )

# with open('train_log.txt', 'w') as f:
#     for i in range(len(train_history.losses)):
#         f.write("%s,%s,%s,%s,%s\n" % (train_history.losses[i],train_history.accuracy[i], train_history.auc[i], train_history.precision[i],train_history.recall[i]))

# model.load_weights("%s/cnn_ring_chk.hdf5"% project_dir)

# for layer in base_model.layers:
#   layer.trainable = True

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-5),  # Low learning rate
#     loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
#     metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
# )

# fine_tuning_csv_logger = CSVLogger('fine_tuning_log_epoch.csv', append=False, separator=';')
# fine_tuning_checkpointer = tf.keras.callbacks.ModelCheckpoint(monitor="val_precision", save_best_only=True, mode="max", filepath="%s/cnn_ring_fine_tuning_chk.hdf5"% project_dir)

# epochs = 50

# model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[fine_tuning_checkpointer, fine_tuning_csv_logger])

# model.load_weights("%s/cnn_ring_fine_tuning_chk.hdf5"% project_dir)

# model.save("%s/cnn_ring.hdf5" % project_dir)
# # model.save_weights("%s/cnn_rings.hdf5"% project_dir)

model = tf.keras.models.load_model("%s/cnn_ring.hdf5"% project_dir)

test_dir = pathlib.Path("%s/data/test/" % project_dir)
test_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

x_test, y_test = [], []

for images, labels in test_ds:
  if(images.shape[0] > 16):
    x_test.append(images.numpy())
    y_test.append(labels.numpy())

y_in = np.empty((0), dtype=int)
y_out = np.empty((0), dtype=int)

for z in range(len(x_test)):
    y_pred = model.predict(x_test[z])
    y_in = np.hstack((y_in, y_test[z]))
    y_out = np.hstack((y_out, y_pred.flatten()))
# RocCurveDisplay.from_predictions(y_in,y_out)
PrecisionRecallDisplay.from_predictions(y_in, y_out)
# print(f1_score(y_in,np.round(y_out)))
plt.xlabel("felidézés")
plt.ylabel("precizitás")
plt.show()

# result = model.evaluate(test_ds)
# print(result)