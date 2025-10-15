# mnist_draw_predict.py
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model

MODEL_PATH = "models/mnist_cnn_best.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training script first.")

model = load_model(MODEL_PATH)

class App:
    def __init__(self, master):
        self.master = master
        master.title("Handwritten Digit Recognizer - Draw and Predict")

        self.canvas_width = 280
        self.canvas_height = 280
        self.bg_color = "black"

        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg=self.bg_color)
        self.canvas.grid(row=0, column=0, columnspan=4, pady=10, padx=10)

        self.label = tk.Label(master, text="Draw a digit (0-9) and click Predict", font=("Helvetica", 14))
        self.label.grid(row=1, column=0, columnspan=4)

        self.predict_button = tk.Button(master, text="Predict", command=self.predict)
        self.predict_button.grid(row=2, column=0, pady=10)

        self.clear_button = tk.Button(master, text="Clear", command=self.clear)
        self.clear_button.grid(row=2, column=1, pady=10)

        self.prob_button = tk.Button(master, text="Show probs", command=self.predict, state='normal')
        self.prob_button.grid(row=2, column=2, pady=10)

        self.quit_button = tk.Button(master, text="Quit", command=master.quit)
        self.quit_button.grid(row=2, column=3, pady=10)

        # For drawing: use PIL image to store strokes
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)  # black background
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

    def paint(self, event):
        # Draw a small circle (brush) on canvas and PIL image
        x, y = event.x, event.y
        r = 12  # brush radius
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill=0)
        self.label.config(text="Draw a digit (0-9) and click Predict")

    def preprocess_image(self, pil_img):
        # pil_img is 280x280 'L' mode (0-255). Convert to 28x28 like MNIST and invert colors
        # Steps: invert (MNIST has white digit on black bg?), resize to 28x28, normalize
        # Our training used white digit on black background; the GUI draws white on black so keep same.
        img = pil_img.resize((28,28), Image.Resampling.LANCZOS)
        arr = np.array(img)  # shape (28,28)
        # Optionally: apply threshold and center-of-mass shifting to center digit (skipped for simplicity)
        arr = arr.astype("float32") / 255.0  # normalize 0-1
        arr = np.expand_dims(arr, axis=-1)  # (28,28,1)
        arr = np.expand_dims(arr, axis=0)   # (1,28,28,1)
        return arr

    def predict(self):
        arr = self.preprocess_image(self.image)
        preds = model.predict(arr)
        class_id = np.argmax(preds, axis=1)[0]
        prob = preds[0, class_id]
        # Show top-3 probabilities
        top3 = np.argsort(preds[0])[::-1][:3]
        prob_text = "Top predictions: " + ", ".join([f"{i}:{preds[0,i]:.2f}" for i in top3])
        self.label.config(text=f"Predicted: {class_id} (confidence: {prob:.2f})\n{prob_text}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
