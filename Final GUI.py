import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import threading
root = tk.Tk()

# Path to the directory where your training and validation data are stored
train_dir = '/home/momen/Desktop/Afnan/FINAL AI/data/train'
validation_dir = '/home/momen/Desktop/Afnan/FINAL AI/data/val'

# Image size used during training
image_size = (100, 100)

# Create data generators
batch_size = 32

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

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

# Load the background image
background_image = Image.open('/home/momen/Desktop/Afnan/FINAL AI/bg2.jpeg')

# Resize the image to match the window size
window_width = 400
window_height = 300
background_image = background_image.resize((window_width, window_height))

# Create a PhotoImage object
background_photo = ImageTk.PhotoImage(image=background_image)

# Tkinter GUI
class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")

        # Set the window size and position it in the center
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")  # Centered on the screen

        # Themed widgets
        self.style = ttk.Style()
        self.style.theme_use('clam')  # You can change the theme if desired

        # Configure style for white and baby blue background
        self.style.configure('TFrame', background='lightblue')  # Frame background color
        self.style.configure('TButton', background='lightblue', foreground='black')  # Button colors
        self.style.configure('TLabel', background='lightblue', foreground='black', font=('Helvetica', 12, 'bold'))  # Label colors

        # Frame to hold the elements
        self.frame = ttk.Frame(root)
        self.frame.pack(expand=True, fill="both")

        # Configure the background of the main frame
        background_label = ttk.Label(self.frame, image=background_photo)
        background_label.place(relwidth=1, relheight=1)
        background_label.image = background_photo  # Reference to keep the image alive

        # GUI elements
        self.load_button = ttk.Button(self.frame, text="Load Image", command=self.load_and_predict_image)
        self.load_button.pack(pady=10)

        self.image_label = ttk.Label(self.frame)
        self.image_label.pack(expand=True)  # Use pack to center the label

        self.prediction_label = ttk.Label(self.frame, text="")
        self.prediction_label.pack()

        self.exit_button = ttk.Button(self.frame, text="Exit", command=root.quit)
        self.exit_button.pack(pady=10)

    def load_and_predict_image(self):
        file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.png *.jpeg")])

        if file_path:
            try:
                # Display loading message
                self.prediction_label.config(text="Loading...")

                # Load and preprocess the image in a separate thread
                threading.Thread(target=self.process_image, args=(file_path,)).start()

            except Exception as e:
                self.prediction_label.config(text="Error: " + str(e))
                self.image_label.config(image=None)

    def process_image(self, file_path):
        img = load_img(file_path, target_size=image_size)
        img = img.resize((100, 100))  # Resize the image to 100x100 pixels
        img_array = img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        predicted_class = tf.argmax(prediction, axis=1)[0]

        predicted_class_name = file_to_class_mapping[file_names[predicted_class]]
        self.prediction_label.config(text="Predicted class: " + predicted_class_name)

        # Display the resized image with fade-in animation
        img = Image.open(file_path)
        img = img.resize((100, 100))  # Resize the image to 100x100 pixels

        img_tk = ImageTk.PhotoImage(img.convert("RGBA"))
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

        self.fade_in_animation()

    def fade_in_animation(self):
        # Simple fade-in animation for the image
        for i in range(10):
            self.frame.after(50)
            self.frame.update()
            self.root.update()
            self.frame.lift()

# Create the main window and run the Tkinter event loop
app = ImageClassifierApp(root)
root.mainloop()
