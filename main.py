# main.py
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

MODEL_PATH = "handwritten.keras"
USE_CNN = True

def build_model():
    if not USE_CNN:
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def load_or_train():
    if os.path.exists(MODEL_PATH):
        print("Found existing model — loading it...")
        return tf.keras.models.load_model(MODEL_PATH)

    print("No model found — training a new one...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    model = build_model()
    model.fit(x_train, y_train, epochs=7, validation_data=(x_test, y_test), verbose=1)
    model.save(MODEL_PATH)
    print("Model trained and saved successfully!")
    return model

def center_of_mass(mask):
    """Compute integer shift to move the mask's COM to image center (14,14)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, 0
    cx = xs.mean()
    cy = ys.mean()
    return int(round(14 - cx)), int(round(14 - cy))

def preprocess(path):
    """
    Convert an arbitrary digit image to MNIST-like:
      - grayscale
      - white digit on black background
      - binarize & crop largest contour
      - resize to fit 20x20, pad to 28x28
      - center by center-of-mass
      - normalize to 0..1, shape (1, 28, 28)
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    # If background is bright, invert so digit is white on black (MNIST style)
    if img.mean() > 127:
        img = 255 - img

    # Light blur + Otsu threshold for a clean mask
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Crop to largest contour (the digit)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found (is the image blank?)")
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    digit = bin_img[y:y+h, x:x+w]

    # Resize to keep aspect and fit into 20x20
    if w > h:
        new_w, new_h = 20, max(1, int(round(h * (20.0 / w))))
    else:
        new_h, new_w = 20, max(1, int(round(w * (20.0 / h))))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to 28x28
    pad_top = (28 - new_h) // 2
    pad_bot = 28 - new_h - pad_top
    pad_left = (28 - new_w) // 2
    pad_right = 28 - new_w - pad_left
    canvas = np.pad(digit, ((pad_top, pad_bot), (pad_left, pad_right)),
                    mode="constant", constant_values=0)

    # Center by center-of-mass
    sx, sy = center_of_mass(canvas > 0)
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    canvas = cv2.warpAffine(canvas, M, (28, 28), borderValue=0)

    # Normalize to 0..1 and add batch dim
    canvas = canvas.astype("float32") / 255.0
    return np.expand_dims(canvas, axis=0)

model = load_or_train()

pic_number = 1
any_found = False
while os.path.isfile(f"digits/digit{pic_number}.png"):
    any_found = True
    path = f"digits/digit{pic_number}.png"
    try:
        x = preprocess(path)
        probs = model.predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        print(f"{os.path.basename(path)} → {pred}  (confidence: {conf:.2%})")

        # Comment out these two lines if you don't want a GUI window
        plt.imshow(x[0], cmap="gray")
        plt.title(f"Pred: {pred} • {conf:.1%}"); plt.axis("off"); plt.show()

    except Exception as e:
        print("Error processing image:", e)
    finally:
        pic_number += 1

if not any_found:
    print("No images matched: digits/digit<N>.png (e.g., digits/digit1.png)")