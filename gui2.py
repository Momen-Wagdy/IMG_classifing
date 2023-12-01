import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Path to the directory where your training and validation data are stored
train_dir = '/home/momen/Desktop/Afnan/FINAL AI/data/train'
validation_dir = '/home/momen/Desktop/Afnan/FINAL AI/data/val'

# Image size used during training
image_size = (100, 100)

# Create data generators
batch_size = 32

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Extract file names and class names
class_names = sorted(os.listdir(train_dir))
file_names = [os.path.splitext(f)[0] for f in train_generator.filenames]

# Map file names to class names
file_to_class_mapping = {file: class_name for file, class_name in zip(file_names, class_names)}

# Load the trained model
model_path = '/home/momen/Desktop/Afnan/FINAL AI/my_model.h5'  # Update with the actual full path to the model file
model = load_model(model_path)

# Tkinter GUI
def load_and_predict_image():
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    
    if file_path:
        try:
            img = load_img(file_path, target_size=image_size)
            img = img.resize((100, 100))  # Resize the image to 100x100 pixels
            img_array = img_to_array(img)
            img_array = tf.expand_dims(img_array, axis=0)
            img_array /= 255.0

            prediction = model.predict(img_array)
            predicted_class = tf.argmax(prediction, axis=1)[0]

            predicted_class_name = file_to_class_mapping[file_names[predicted_class]]
            prediction_label.config(text="Predicted class: " + predicted_class_name)
            
            # Display the resized image
            img = Image.open(file_path)
            img = img.resize((100, 100))  # Resize the image to 100x100 pixels
            img = ImageTk.PhotoImage(img)
            image_label.config(image=img)
            image_label.image = img
            
        except Exception as e:
            prediction_label.config(text="Error: " + str(e))
            image_label.config(image=None)


# Create the main window
root = tk.Tk()
root.title("Image Classifier")

# Create GUI elements
load_button = tk.Button(root, text="Load Image", command=load_and_predict_image)
load_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

prediction_label = tk.Label(root, text="")
prediction_label.pack()

exit_button = tk.Button(root, text="Exit", command=root.quit)
exit_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
