Face Recognition using LBPH (Local Binary Patterns Histogram)
This repository provides a face recognition system implemented using the Local Binary Patterns Histogram (LBPH) algorithm. LBPH is widely used for facial recognition tasks because of its simplicity and effectiveness in recognizing faces with minimal computational power.

Features
Face Data Collection: Capture and store face images from the camera.
Model Training: Train the LBPH model with the captured face data.
Face Recognition: Test the trained model to recognize faces in real time.
Project Structure
datacollect.py: Captures images from the camera, labels, and stores them for training.
trainingdemo.py: Trains an LBPH face recognizer model with the collected images.
testmodel.py: Tests the model’s accuracy on new or existing faces in real time.
Installation
Clone this repository:
bash
Copy code
git clone https://github.com/DSHREEKARTIK/face_recognition_lbph.git
Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Note: Ensure you have OpenCV installed for image processing.
Usage
Data Collection: Run datacollect.py to capture face data.
bash
Copy code
python datacollect.py
Model Training: Train the model using trainingdemo.py.
bash
Copy code
python trainingdemo.py
Face Recognition: Test the model on new faces using testmodel.py.
bash
Copy code
python testmodel.py
Requirements
Python 3.x
OpenCV
Numpy
How it Works
The LBPH algorithm breaks down an image into small grids, processes each with local binary patterns, and creates histograms to classify faces. This method allows for effective recognition even in lower-quality images.
