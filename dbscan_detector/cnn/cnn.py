from email.mime import base
import shutil
import tensorflow as tf
from keras import models, layers
from keras_preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
import numpy as np
import os
import pathlib
from PIL import Image
from keras.callbacks import CSVLogger

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))

def image_validation():
  project_dir = os.path.dirname(__file__)
  spirals = os.listdir("%s/data/spiral" % project_dir)
  ellipticals = os.listdir("%s/data/elliptical" % project_dir)
  for s in spirals:
    img = Image.open("%s/data/spiral/%s" % (project_dir, s))
    try:
        img.verify()
    except Exception:
        print('Invalid image %s' % s)

  for e in ellipticals:
    img = Image.open("%s/data/elliptical/%s" % (project_dir, e))
    try:
        img.verify()
    except Exception:
        print('Invalid image %s' % s)

# image_validation()

project_dir = os.path.dirname(__file__)
data_dir = pathlib.Path("%s/data/" % project_dir)
solution_dir = pathlib.Path(project_dir).parent

# spiral_files = os.listdir("%s/images/spirals/clean" % solution_dir)
# elliptical_files = os.listdir("%s/images/ellipticals/clean" % solution_dir)

# train_spiral_files = spiral_files[0:int(len(spiral_files)*0.7)]
# val_spiral_files = spiral_files[int(len(spiral_files)*0.7):int(len(spiral_files)*0.9)]
# test_spiral_files = spiral_files[int(len(spiral_files)*0.9):int(len(spiral_files))]

# train_elliptical_files = elliptical_files[0:int(len(elliptical_files)*0.7)]
# val_elliptical_files = elliptical_files[int(len(elliptical_files)*0.7):int(len(elliptical_files)*0.9)]
# test_elliptical_files = elliptical_files[int(len(elliptical_files)*0.9):int(len(elliptical_files))]

# for f in train_spiral_files:
#   shutil.copy("%s/images/spirals/clean/%s" % (solution_dir, f), "%s/train/spiral/%s" % (data_dir, f))
# for f in val_spiral_files:
#   shutil.copy("%s/images/spirals/clean/%s" % (solution_dir, f), "%s/val/spiral/%s" % (data_dir, f))
# for f in test_spiral_files:
#   shutil.copy("%s/images/spirals/clean/%s" % (solution_dir, f), "%s/test/spiral/%s" % (data_dir, f))

# for f in train_elliptical_files:
#   shutil.copy("%s/images/ellipticals/clean/%s" % (solution_dir, f), "%s/train/elliptical/%s" % (data_dir, f))
# for f in val_elliptical_files:
#   shutil.copy("%s/images/ellipticals/clean/%s" % (solution_dir, f), "%s/val/elliptical/%s" % (data_dir, f))
# for f in test_elliptical_files:
#   shutil.copy("%s/images/ellipticals/clean/%s" % (solution_dir, f), "%s/test/elliptical/%s" % (data_dir, f))

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
    layers.RandomTranslation(height_factor=(-0.1,0.1),width_factor=(-0.1,0.1), fill_mode="nearest")
  ]
)

datagen = ImageDataGenerator(
        # rotation_range=45,
        width_shift_range=0.4,
        height_shift_range=0.4,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True,
        # vertical_flip=True,
        # brightness_range=(0.3, 1.0),
        fill_mode='constant', cval=0)

# train_generator = datagen.flow_from_directory(
#         train_dir,  # this is the target directory
#         target_size=(128, 128),  # all images will be resized to 150x150
#         batch_size=batch_size,
#         class_mode='sparse')  # since we use binary_crossentropy loss, we need binary labels

# validation_generator = datagen.flow_from_directory(
#         val_dir,
#         target_size=(128, 128),
#         batch_size=batch_size,
#         class_mode='sparse')

test_dir = pathlib.Path("%s/data/test/" % project_dir)
test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='sparse')

# model = tf.keras.Sequential([
#   layers.Rescaling(1./255, input_shape=(128, 128, 3)),
#   data_augmentation,
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.BatchNormalization(),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.BatchNormalization(),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.BatchNormalization(),
#   layers.MaxPooling2D(),
#   layers.Conv2D(128, 3, padding='same', activation='relu'),
#   layers.BatchNormalization(),
#   layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dense(256, activation='relu'),
#   layers.Dropout(0.1),
#   layers.Dense(3)
# ])


base_model = tf.keras.applications.ResNet50(weights = 'imagenet', include_top = False, input_shape = (128,128,3))
# for layer in base_model.layers:
#   layer.trainable = True

inp = layers.Input((128,128,3))
x = data_augmentation(inp)
x = base_model(x)
# x = layers.Flatten()(x)
# x = layers.Dense(128, activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)
predictions = layers.Dense(3)(x)

model = tf.keras.Model(inputs = inp, outputs = predictions)

model_checkpointer = tf.keras.callbacks.ModelCheckpoint(monitor="val_accuracy", save_best_only=True, mode="max", filepath="%s/cnn_robust_test_chk.hdf5"% project_dir)

model.compile(optimizer='SGD',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics=['accuracy'])

# model.summary()
# plot_model(model, to_file='%s/CNN_main_cat_plot.png' % project_dir, show_shapes=True, show_layer_names=True, show_layer_activations=True)
# train_csv_logger = CSVLogger('train_log_epoch.csv', append=False, separator=';')
def lr_schedule(epoch):
    return 0.01*(1-0.9*epoch/4)

train_history = LossHistory()

epochs=1
# history = model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs,
#   callbacks=[model_checkpointer, tf.keras.callbacks.LearningRateScheduler(lr_schedule)]#, train_history, train_csv_logger]
# )

# with open('train_log.txt', 'w') as f:
#     for i in range(len(train_history.losses)):
#         f.write("%s,%s\n" % (train_history.losses[i],train_history.accuracy[i]))
model = tf.keras.models.load_model("%s/cnn_robust_test.hdf5"% project_dir)
# model.load_weights("%s/cnn_robust_test_chk.hdf5"% project_dir)

# model.save_weights("%s/cnn_weights.hdf5"% project_dir)
# model.save("%s/cnn_robust_test.hdf5" % project_dir)

# model.load_weights("%s/cnn_main.hdf5"% project_dir)

# for layer in base_model.layers:
#   layer.trainable = True

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-5),  # Low learning rate
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
#     metrics=['accuracy']
# )

# model_earlystop = tf.keras.callbacks.EarlyStopping(
#     monitor="val_accuracy",
#     min_delta=0,
#     patience=5,
#     verbose=0,
#     mode="auto",
#     baseline=None,
#     restore_best_weights=True,
# )

# epochs = 10
# model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[model_earlystop])

# model.save_weights("%s/cnn_main_weights.hdf5"% project_dir)
# model.save("%s/cnn_main.hdf5" % project_dir)

# model.load_weights("%s/cnn_test.hdf5"% project_dir)
# model = tf.keras.models.load_model("%s/cnn_main.hdf5"% project_dir)

test_dir = pathlib.Path("%s/data/test/" % project_dir)
test_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

result = model.evaluate(test_ds)
print(result)