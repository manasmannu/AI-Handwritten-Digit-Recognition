🧠 Handwritten Digit Recognition

A simple TensorFlow/Keras project that trains a neural network on the MNIST dataset to recognize handwritten digits (0–9). It uses OpenCV for image preprocessing and Matplotlib for visualizing predictions.

⸻

⚙️ Setup Instructions

1️⃣ Clone or Download the Project

git clone <your-repo-url>
cd AI_Mini_Project-main

2️⃣ Create and Activate a Virtual Environment

Mac/Linux:

python3 -m venv venv
source venv/bin/activate

Windows (PowerShell):

python -m venv venv
venv\Scripts\activate

When activated, your terminal prompt should show (venv).

⸻

3️⃣ Install Required Packages

Make sure you’re inside the virtual environment before running this:

pip install -r requirements.txt

If you don’t have a requirements.txt file yet, create one with this content:

tensorflow
opencv-python
matplotlib
numpy

Then install it using the same command above.

⸻

4️⃣ Run the Program

To train the model (if no saved model exists) and predict digits:

python main.py

If you want to retrain the model from scratch:

rm handwritten.keras # macOS/Linux

# or

Remove-Item handwritten.keras # Windows PowerShell
python main.py

⸻

5️⃣ Directory Structure

AI_Mini_Project-main/
│
├── main.py # Main script (training + prediction)
├── handwritten.keras # Saved model file
├── digits/ # Folder containing digit images
│ ├── digit1.png
│ ├── digit2.png
│ └── ...
├── requirements.txt
└── README.md

⸻

6️⃣ Example Output

When you run the program, you’ll see predictions printed in the terminal like:

digit1.png → 3 (confidence: 98.12%)
digit2.png → 8 (confidence: 94.77%)

Each image will also be displayed in a pop-up window with its predicted label.

⸻

7️⃣ Common Issues

SSL error when downloading MNIST:
If you get a certificate error:

/Applications/Python\ 3.x/Install\ Certificates.command

This fixes SSL issues on macOS.

GUI not showing:
If you’re running on macOS and matplotlib windows don’t open, set:

matplotlib.use("Qt5Agg")

inside main.py (already included).

⸻

🔁 8️⃣ Retrain Option (via Command Flag)

If you want a simpler way to retrain, modify your script to include this near the top:

import argparse, os, shutil

parser = argparse.ArgumentParser()
parser.add_argument('--retrain', action='store_true', help='delete saved model and train from scratch')
args = parser.parse_args()

MODEL_PATH = 'handwritten.keras'
if args.retrain and os.path.exists(MODEL_PATH):
os.remove(MODEL_PATH)

Now you can retrain anytime using:

python main.py --retrain

This will delete the old model and train a fresh one automatically.

⸻

✅ Done!

You now have a working handwritten digit recognition model. You can replace or add new images in the digits/ folder to test custom inputs.
