import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define your training, validation, and test directories
train_dir = '/home/momen/Desktop/Afnan/FINAL AI/data/train'
validation_dir = '/home/momen/Desktop/Afnan/FINAL AI/data/val'
test_dir = '/home/momen/Desktop/Afnan/FINAL AI/data/test'

# Define hyperparameters
batch_size = 32
image_size = (100, 100)  # Adjust the size according to your model's input size

# Create data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1.0/255)
validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Get the number of classes dynamically
num_classes = len(train_generator.class_indices)

# Build the neural network model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')  # Updated to use dynamically determined number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model for exactly n epochs
model.fit(train_generator, validation_data=validation_generator, epochs=30)

# Save the entire model
model.save('my_model.h5')
print("Model saved.")

# Now, let's use the trained model to classify user-provided images
from tensorflow.keras.preprocessing import image
import numpy as np

def classify_user_image(image_path, model):
    img = image.load_img(image_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    return predicted_class

while True:
    user_image_path = input("Enter the path to the user-provided image (or 'exit' to quit): ")

    if user_image_path.lower() == 'exit':
        break

    try:
        predicted_class = classify_user_image(user_image_path, model)
        print("Predicted class index:", predicted_class)
    except Exception as e:
        print("Error:", str(e))
