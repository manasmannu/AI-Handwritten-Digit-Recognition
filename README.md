# Handwritten Digit Recognition

A simple TensorFlow/Keras project that trains a neural network on the MNIST dataset to recognize handwritten digits (0–9). It uses OpenCV for image preprocessing and Matplotlib for visualizing predictions.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)
![Keras](https://img.shields.io/badge/Library-Keras-red)
![Matplotlib](https://img.shields.io/badge/Visualization-Matplotlib-yellow)
![MNIST](https://img.shields.io/badge/Dataset-MNIST-purple)

### Setup Instructions

### Clone or Download the Project

```bash
git clone (https://github.com/manasmannu/AI-Handwritten-Digit-Recognition)
cd <folder_name>
```

### Create and Activate a Virtual Environment

Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (PowerShell):

```bash
python -m venv venv
venv\Scripts\activate
```

When activated, your terminal prompt should show (venv).

### Install Required Packages

Make sure you’re inside the virtual environment before running this:

```bash
pip install -r requirements.txt
```

### Run the Program

To train the model (if no saved model exists) and predict digits:

```bash
python main.py
```

If you want to retrain the model from scratch:

rm handwritten.keras # macOS/Linux

# or

Remove-Item handwritten.keras # Windows PowerShell

Then rerun the below command

```bash
python main.py
```

### Directory Structure

<p align="left">
  <img src="https://github.com/user-attachments/assets/b0475bb0-c22a-483a-b036-b2e3975bdaf6" 
       alt="Directory" width="45%">
</p>

### Example Output

When you run the program, you’ll see predictions printed in the terminal like:

digit1.png → 3 (confidence: 98.12%)
digit2.png → 8 (confidence: 94.77%)

Each image will also be displayed in a pop-up window with its predicted label.

### Demo Screenshot

<p align="center">
  <img src="https://github.com/user-attachments/assets/7b2abe18-84fd-413e-9c07-cdde964b9260" 
       alt="Pop-up Demo" width="65%">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/9b54e16a-0105-4b2d-a12a-76484261f757" 
       alt="Pop-up Demo" width="65%">
</p>

You can upload more "png" format images here to test the model

<p align="center">
  <img src="https://github.com/user-attachments/assets/8b82929d-7acd-4efe-93ae-b75850a75491" 
       alt="Digits Folder" width="45%">
</p>

### Common Issues

SSL error when downloading MNIST:
If you get a certificate error:

run this on terminal

```bash
/Applications/Python\ 3.x/Install\ Certificates.command
```

This fixes SSL issues on macOS.

### Done

You now have a working handwritten digit recognition model. You can replace or add new images in the digits/ folder to test custom inputs. Make sure they are in "png" format.
