import os
import threading
import flet as ft
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class ImageClassifierApp:
    def __init__(self):
        self.root = ft.App(target=self.main, view=ft.WEB_BROWSER, port=33806)

        # Path to the directory where your training and validation data are stored
        self.train_dir = '/home/momen/Desktop/Afnan/FINAL AI/data/train'
        self.validation_dir = '/home/momen/Desktop/Afnan/FINAL AI/data/val'

        # Image size used during training
        self.image_size = (100, 100)

        # Create data generators
        self.batch_size = 32

        self.train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
        self.validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

        self.train_generator = self.train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical'
        )

        # Extract file names and class names
        self.class_names = sorted(os.listdir(self.train_dir))
        self.file_names = [os.path.splitext(f)[0] for f in self.train_generator.filenames]

        # Map file names to class names
        self.file_to_class_mapping = {file: class_name for file, class_name in zip(self.file_names, self.class_names)}

        # Load the trained model
        self.model_path = '/home/momen/Desktop/Afnan/FINAL AI/my_model.h5'
        self.model = load_model(self.model_path)

    def main(self, page: ft.Page):
        page.title = "Image Classifier"
        page.window_height = 300
        page.window_width = 400
        page.window_resizable = False

        # Load the background image
        background_image = Image.open('/home/momen/Desktop/Afnan/FINAL AI/bg2.jpeg')
        background_image = background_image.resize((400, 300))
        background_photo = ImageTk.PhotoImage(image=background_image)

        # Frame to hold the elements
        frame = ft.Frame(page)
        frame.pack(expand=True, fill="both")

        # Configure the background of the main frame
        background_label = ft.Label(frame, image=background_photo)
        background_label.place(relwidth=1, relheight=1)
        background_label.image = background_photo

        # GUI elements
        load_button = ft.Button(frame, text="Load Image", on_click=self.load_and_predict_image)
        load_button.pack(pady=10)

        image_label = ft.Label(frame, id="image_label")
        image_label.pack(expand=True)

        prediction_label = ft.Label(frame, text="", id="prediction_label")
        prediction_label.pack()

        exit_button = ft.Button(frame, text="Exit", on_click=page.close)
        exit_button.pack(pady=10)

    def load_and_predict_image(self, e):
        file_path = ft.file_dialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.png *.jpeg")])

        if file_path:
            try:
                prediction_label = e.page.get_element_by_id("prediction_label")
                prediction_label.text = "Loading..."
                threading.Thread(target=self.process_image, args=(e, file_path)).start()

            except Exception as ex:
                prediction_label = e.page.get_element_by_id("prediction_label")
                prediction_label.text = f"Error: {str(ex)}"
                image_label = e.page.get_element_by_id("image_label")
                image_label.image = None

    def process_image(self, e, file_path):
        img = load_img(file_path, target_size=self.image_size)
        img = img.resize((100, 100))  # Resize the image to 100x100 pixels
        img_array = img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = self.model.predict(img_array)
        predicted_class = tf.argmax(prediction, axis=1)[0]

        predicted_class_name = self.file_to_class_mapping[self.file_names[predicted_class]]
        prediction_label = e.page.get_element_by_id("prediction_label")
        prediction_label.text = "Predicted class: " + predicted_class_name

        # Display the resized image with fade-in animation
        img = Image.open(file_path)
        img = img.resize((100, 100))  # Resize the image to 100x100 pixels

        img_tk = ImageTk.PhotoImage(img.convert("RGBA"))
        image_label = e.page.get_element_by_id("image_label")
        image_label.image = img_tk
        image_label.update()

        self.fade_in_animation()

    def fade_in_animation(self):
        # Flet doesn't require a separate animation function for the fade-in effect
        pass

# Run the Flet app
if __name__ == "__main__":
    app = ImageClassifierApp()
    app.root.run()
