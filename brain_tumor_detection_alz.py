import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import splitfolders
import cv2
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight


# 1. CONFIGURATION & GPU SETUP
RAW_DATASET_PATH = "/mnt/e/D-Temp/5thSem/Dataset"         # Your original folder
SPLIT_DATASET_PATH = "/mnt/e/D-Temp/5thSem/td" # Where split data goes

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
NUM_CLASSES = 7

def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU Detected: {len(gpus)} device(s) active.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected. Training will be slow.")

setup_gpu()


# 2. PREPROCESSING A: DATA SPLITTING
if not os.path.exists(SPLIT_DATASET_PATH):
    print(f"[INFO] Splitting data into Train (70%), Val (15%), Test (15%)...")
    splitfolders.ratio(
        RAW_DATASET_PATH, 
        output=SPLIT_DATASET_PATH, 
        seed=1337, 
        ratio=(.7, .15, .15), 
        group_prefix=None
    )
    print("Data Split Complete!")
else:
    print("[INFO] Split dataset found. Skipping split step.")


# 3. PREPROCESSING B: GENERATORS & AUGMENTATION
print("\n[INFO] Creating Data Generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalization
    rotation_range=10,        # Augmentation
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(SPLIT_DATASET_PATH, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = test_val_datagen.flow_from_directory(
    os.path.join(SPLIT_DATASET_PATH, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_val_datagen.flow_from_directory(
    os.path.join(SPLIT_DATASET_PATH, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())


# 4. PREPROCESSING C: CLASS BALANCING
print("\n[INFO] Computing Class Weights for Balancing...")

# We extract the class labels from the generator
train_labels = train_generator.classes 

# Compute weights: 'balanced' automatically calculates weights inversely proportional to class frequencies
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

# Convert to dictionary format required by Keras
class_weights_dict = dict(enumerate(class_weights))

print(f"Class Weights Computed: {class_weights_dict}")
# This dictionary will be passed to model.fit() later


# 5. CUSTOM CNN ARCHITECTURE (NeuroDetect-7)
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Entry
    x = layers.Conv2D(32, (3, 3), strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Deep Residual Blocks
    previous_block_activation = x
    for size in [64, 128, 256]:
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(size, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(size, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

        residual = layers.Conv2D(size, (1, 1), strides=2, padding='same')(previous_block_activation)
        x = layers.add([x, residual])
        previous_block_activation = x

    # Exit & Classifier
    x = layers.SeparableConv2D(512, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Named layer for Grad-CAM
    x = layers.Conv2D(512, (3, 3), padding='same', name='last_conv_layer')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="NeuroDetect-7")

model = build_model(input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)


# 6. TRAINING (With Class Weights)
checkpoint = ModelCheckpoint('neurodetect_best.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

print("\n[INFO] Starting Training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint, reduce_lr],
    class_weight=class_weights_dict  # <--- HERE IS THE BALANCING MAGIC
)


# 7. FINAL EVALUATION (Test Set)
print("\n[INFO] Loading Best Model for Testing...")
best_model = tf.keras.models.load_model('neurodetect_best.keras')

print("[INFO] Evaluating on Test Set...")
loss, acc, prec, rec = best_model.evaluate(test_generator)
print(f"\n🚀 Final Test Accuracy: {acc*100:.2f}%")

# Classification Report
test_generator.reset()
Y_pred = best_model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(test_generator.classes, y_pred, target_names=class_names))


# 8. VISUALIZATION (History & Grad-CAM)
def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy
    ax[0].plot(history.history['accuracy'], label='Train')
    ax[0].plot(history.history['val_accuracy'], label='Val')
    ax[0].set_title('Model Accuracy')
    ax[0].legend()

    # Loss
    ax[1].plot(history.history['loss'], label='Train')
    ax[1].plot(history.history['val_loss'], label='Val')
    ax[1].set_title('Model Loss')
    ax[1].legend()
    plt.show()

plot_history(history)

# Grad-CAM Function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Test Grad-CAM
print("\n[INFO] Generating Grad-CAM for Test Sample...")
sample_images, sample_labels = next(test_generator)
sample_img = sample_images[0]
sample_img_batch = np.expand_dims(sample_img, axis=0)

heatmap = make_gradcam_heatmap(sample_img_batch, best_model, 'last_conv_layer')

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(sample_img)
plt.title("Test Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sample_img)
plt.imshow(cv2.resize(heatmap, (224, 224)), cmap='jet', alpha=0.4)
plt.title("Grad-CAM Overlay")
plt.axis('off')
plt.show()

"""
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import splitfolders
import cv2
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight


# 1. CONFIGURATION & GPU SETUP
RAW_DATASET_PATH = './Dataset'         
SPLIT_DATASET_PATH = './Dataset_Split' 

# Hyperparameters (Optimized for Transfer Learning)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # Reduced to 32 for better generalization
EPOCHS = 25      # EfficientNet converges faster, so 20 is usually enough
LEARNING_RATE = 0.0001
NUM_CLASSES = 7

def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU Detected: {len(gpus)} device(s) active.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected. Training will be slow.")

setup_gpu()


# 2. PREPROCESSING A: ROBUST DATA SPLITTING
# Check if 'train' exists to confirm split was successful previously
if not os.path.exists(os.path.join(SPLIT_DATASET_PATH, 'train')):
    print(f"[INFO] Splitting data into Train (70%), Val (15%), Test (15%)...")
    # Clean up any potential corrupt folders
    if os.path.exists(SPLIT_DATASET_PATH):
        shutil.rmtree(SPLIT_DATASET_PATH)
    
    splitfolders.ratio(
        RAW_DATASET_PATH, 
        output=SPLIT_DATASET_PATH, 
        seed=1337, 
        ratio=(.7, .15, .15), 
        group_prefix=None
    )
    print("Data Split Complete!")
else:
    print("[INFO] Split dataset found. Skipping split step.")


# 3. PREPROCESSING B: GENERATORS
print("\n[INFO] Creating Data Generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,        # Slightly increased for robust feature learning
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(SPLIT_DATASET_PATH, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

print(f"Classes found: {train_generator.class_indices.keys()}")
if train_generator.num_classes != NUM_CLASSES:
    print(f"❌ ERROR: Expected {NUM_CLASSES} classes, but found {train_generator.num_classes}.")
    print("Please delete the 'Dataset_Split' folder and check your source 'Dataset' folder.")
    exit() # Stop the script so you don't get the confusing Shape Error

val_generator = test_val_datagen.flow_from_directory(
    os.path.join(SPLIT_DATASET_PATH, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_val_datagen.flow_from_directory(
    os.path.join(SPLIT_DATASET_PATH, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())


# 4. PREPROCESSING C: CLASS BALANCING
print("\n[INFO] Computing Class Weights...")
train_labels = train_generator.classes 
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights_dict = dict(enumerate(class_weights))
print(f"Weights: {class_weights_dict}")


# 5. MODEL: TRANSFER LEARNING (EfficientNetB0)
def build_transfer_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # 1. Load EfficientNetB0 Pre-trained on ImageNet
    # include_top=False means we cut off the final classification layer
    base_model = EfficientNetB0(
        include_top=False, 
        weights='imagenet', 
        input_tensor=inputs
    )

    # 2. Fine-Tuning Strategy
    # We freeze the bottom layers (generic features) and unfreeze the top (specific features)
    base_model.trainable = True
    for layer in base_model.layers[:-20]: # Freeze all except last 20 layers
        layer.trainable = False

    x = base_model.output

    # 3. Custom Top Layers (The "Head")
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    
    # Dense layer with Swish activation (Better than ReLU for deep nets)
    x = layers.Dense(256, activation='swish')(x) 
    x = layers.Dropout(0.5)(x) # Strong dropout to prevent overfitting
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="EfficientNet_Transfer")
    return model

model = build_transfer_model(input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES)

# Label Smoothing: Helps the model generalize better by preventing "overconfidence"
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=loss_fn,
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

model.summary()


# 6. TRAINING
checkpoint = ModelCheckpoint('neurodetect_efficientnet.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

print("\n[INFO] Starting Transfer Learning...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint, reduce_lr],
    class_weight=class_weights_dict
)


# 7. FINAL EVALUATION
print("\n[INFO] Loading Best Model for Testing...")
best_model = tf.keras.models.load_model('neurodetect_efficientnet.keras')

print("[INFO] Evaluating on Test Set...")
loss, acc, prec, rec = best_model.evaluate(test_generator)
print(f"\nFinal Test Accuracy: {acc*100:.2f}%")

# Classification Report
test_generator.reset()
Y_pred = best_model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(test_generator.classes, y_pred, target_names=class_names))


# 8. VISUALIZATION
def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(history.history['accuracy'], label='Train')
    ax[0].plot(history.history['val_accuracy'], label='Val')
    ax[0].set_title('Accuracy')
    ax[0].legend()

    ax[1].plot(history.history['loss'], label='Train')
    ax[1].plot(history.history['val_loss'], label='Val')
    ax[1].set_title('Loss')
    ax[1].legend()
    plt.show()

plot_history(history)

# Grad-CAM specific for EfficientNet
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='top_activation', pred_index=None):
    # 'top_activation' is the standard name for the last conv layer in EfficientNetB0
    
    # We need to find the EfficientNet part inside our custom model
    # Our model structure is: Input -> EfficientNet -> Pooling -> Dense
    # So we act on the model directly, but we need to target the layer name correctly.
    
    grad_model = models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Test Grad-CAM
print("\n[INFO] Generating Grad-CAM for Test Sample...")
try:
    sample_images, sample_labels = next(test_generator)
    sample_img = sample_images[0]
    sample_img_batch = np.expand_dims(sample_img, axis=0)

    # Note: 'top_activation' is the layer name in standard EfficientNetB0
    heatmap = make_gradcam_heatmap(sample_img_batch, best_model, 'top_activation')

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(sample_img)
    plt.title("Test Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sample_img)
    plt.imshow(cv2.resize(heatmap, (224, 224)), cmap='jet', alpha=0.4)
    plt.title("Grad-CAM Overlay")
    plt.axis('off')
    plt.show()
except Exception as e:
    print(f"⚠️ Could not generate Grad-CAM: {e}")
    print("Tip: Check model.summary() to find the exact name of the last Conv layer.")
"""