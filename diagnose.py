import os, warnings, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"TF: {tf.__version__}")

# Config
DATASET_PATH = r"C:\Users\Asus\Desktop\Heart Disease\DATASET"
TRAIN_DIR = os.path.join(DATASET_PATH, "Train")
TEST_DIR  = os.path.join(DATASET_PATH, "Test")
IMG_SIZE    = (224, 224)
IMG_SHAPE   = (224, 224, 3)
BATCH_SIZE  = 32
NUM_CLASSES = 4
CLASS_NAMES = ["Abnormal", "History_MI", "MI", "Normal"]

print("Train dir exists:", os.path.exists(TRAIN_DIR))
print("Test dir exists:", os.path.exists(TEST_DIR))

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest',
    validation_split=0.15,
)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

print("Creating train generator...")
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training', shuffle=True, seed=SEED,
)
print("Creating val generator...")
val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', shuffle=False, seed=SEED,
)
print("Creating test generator...")
test_gen = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False,
)

print(f"Train: {train_gen.samples}, Val: {val_gen.samples}, Test: {test_gen.samples}")

# Test each model build
print("\nTesting Custom CNN build...")
inp = layers.Input(shape=IMG_SHAPE)
x = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
x = layers.GlobalAveragePooling2D()(x)
out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
cnn = models.Model(inp, out)
print("CNN OK, params:", cnn.count_params())

print("\nTesting ResNet50 build...")
try:
    base = ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
    print("ResNet50 base OK, params:", base.count_params())
except Exception as e:
    print("ResNet50 FAILED:", e)

print("\nTesting EfficientNetB0 build...")
try:
    base2 = EfficientNetB0(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
    print("EfficientNetB0 base OK, params:", base2.count_params())
except Exception as e:
    print("EfficientNetB0 FAILED:", e)

print("\nTesting MobileNetV2 build...")
try:
    base3 = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
    print("MobileNetV2 base OK, params:", base3.count_params())
except Exception as e:
    print("MobileNetV2 FAILED:", e)

print("\nAll tests complete!")
