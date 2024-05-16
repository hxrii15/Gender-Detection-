!pip install tensorflow
!pip install cvlib
!pip install opencv-python
 """Install the dependencies seprat tab"""

from google.colab import drive
drive.mount('/content/drive') 
"""import the drive in seprate tab"""

from google.colab.patches import cv2_imshow
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

# Load model
model = load_model('/content/drive/MyDrive/ Gender_detection.model')

# Specify image file path
# Update with your image file path
image_path = '/content/drive/MyDrive/Sample /TVD.jpg'

# Read image from file
frame = cv2.imread(image_path)
# Apply face detection`
face, confidence = cv.detect_face(frame)

# Define class labels
classes = ['Male', 'Female']

# Loop through detected faces
for idx, f in enumerate(face):
    # Get corner points of face rectangle
    startX, startY = f[0], f[1]
    endX, endY = f[2], f[3]

    # Draw rectangle over face
    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Crop the detected face region
    face_crop = np.copy(frame[startY:endY, startX:endX])

    if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
        continue

    # Preprocessing for gender detection model
    face_crop = cv2.resize(face_crop, (96, 96))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = img_to_array(face_crop)
    face_crop = np.expand_dims(face_crop, axis=0)

    # Apply gender detection on face
    conf = model.predict(face_crop)[0]
    idx = np.argmax(conf)
    label = classes[idx]
    label = "{}: {:.2f}%".format(label, conf[idx] * 100)
    Y = startY - 10 if startY - 10 > 10 else startY + 10

    # Write label and confidence above face rectangle
    cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

# Display output
cv2_imshow(frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
