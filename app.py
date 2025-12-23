import tensorflow as tf
import numpy as np
from tkinter import Tk, Canvas, Button
from PIL import Image, ImageOps, ImageDraw
import cv2

# Load MNIST data
def get_mnist_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train, y_train, x_test, y_test)

# Data Augmentation function
def augment_data(x_train, y_train):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    datagen.fit(x_train.reshape(-1, 28, 28, 1))
    return datagen

# Train model
def train_model(x_train, y_train, x_test, y_test):
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train with Data Augmentation
    datagen = augment_data(x_train, y_train)
    model.fit(datagen.flow(x_train, y_train, batch_size=32), 
              validation_data=(x_test, y_test), epochs=10)

    return model

# Improved Preprocessing for input images
def preprocess_image(image):
    img = image.resize((28, 28), Image.Resampling.LANCZOS)
    img = ImageOps.invert(img)
    
    img_data = np.array(img)
    
    # Remove noise using OpenCV
    img_data = cv2.GaussianBlur(img_data, (5, 5), 0)
    _, img_data = cv2.threshold(img_data, 50, 255, cv2.THRESH_BINARY)

    # Crop to digit region
    coords = np.column_stack(np.where(img_data > 0))
    if coords.any():
        top_left = coords.min(axis=0)
        bottom_right = coords.max(axis=0)
        cropped = img_data[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
        img = Image.fromarray(cropped).resize((28, 28), Image.Resampling.LANCZOS)
    
    img_data = np.array(img) / 255.0
    return img_data.reshape(28, 28, 1)

# Prediction function
def predict(model, img):
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    return np.argmax(predictions), np.max(predictions)

# Tkinter GUI for Drawing
class DrawingApp:
    def __init__(self, model):
        self.model = model
        self.window = Tk()
        self.window.title("Digit Recognizer")
        self.canvas = Canvas(self.window, width=280, height=280, bg='black')
        self.canvas.pack()

        self.predict_button = Button(self.window, text="Predict", command=self.predict_digit)
        self.predict_button.pack()

        self.clear_button = Button(self.window, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.canvas.bind('<B1-Motion>', self.draw)
        self.image = Image.new("L", (280, 280), color=0)
        self.draw_context = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        r = 10  
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
        self.draw_context.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new("L", (280, 280), color=0)
        self.draw_context = ImageDraw.Draw(self.image)

    def predict_digit(self):
        img_data = preprocess_image(self.image)
        predicted_digit, confidence = predict(self.model, img_data)
        print(f"Predicted Digit: {predicted_digit} (Confidence: {confidence:.2f})")
        self.canvas.create_text(140, 140, text=str(predicted_digit), fill='red', font=('Helvetica', 40))

    def run(self):
        self.window.mainloop()

# Main function
def main():
    try:
        model = tf.keras.models.load_model('digit_recognizer_model.h5')
        print("Loaded saved model.")
    except:
        print("Training a new model...")
        x_train, y_train, x_test, y_test = get_mnist_data()
        model = train_model(x_train, y_train, x_test, y_test)
        model.save('digit_recognizer_model.h5')
        print("Model saved.")

    app = DrawingApp(model)
    app.run()

if __name__ == "__main__":
    main()