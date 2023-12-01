import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Path to the directory where your training and validation data are stored
train_dir = 'C:\\Users\\kimo store\\PycharmProjects\\Final_AI\\data\\train'
validation_dir = 'C:\\Users\\kimo store\\PycharmProjects\\Final_AI\\data\\val'

# Image size used during training
image_size = (224, 224)

# Create data generators
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1.0/255)
validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Extract file names and class names
class_names = sorted(os.listdir(train_dir))  # Get the list of class names based on subdirectories in the train directory
file_names = [os.path.splitext(f)[0] for f in train_generator.filenames]  # Get the list of file names without extensions

# Map file names to class names
file_to_class_mapping = {file: class_name for file, class_name in zip(file_names, class_names)}


# Load the trained model
model = tf.keras.models.load_model('my_model_new1.keras')

while True:
    user_image_path = input("Enter the path to the user-provided image (or 'exit' to quit): ")

    if user_image_path.lower() == 'exit':
        break

    try:
        img = tf.keras.preprocessing.image.load_img(user_image_path, target_size=image_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        predicted_class = tf.argmax(prediction, axis=1)[0]

        predicted_class_name = file_to_class_mapping[file_names[predicted_class]]  # Convert to numpy array before indexing
        print("Predicted class:", predicted_class_name)
    except Exception as e:
        print("Error:", str(e))