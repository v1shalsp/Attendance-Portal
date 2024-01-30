import os
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm
from types import MethodType

from sklearn.metrics import average_precision_score

### helper function
def encode(img):
 res = resnet(torch.Tensor(img))
 return res

# Custom face detection method
def detect_box(self, img, save_path=None):
  batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
 
  # Select Faces (Skipping small ones)
  if not self.keep_all:
    small_face_indices = [i for i, box in enumerate(batch_boxes) if box[2] - box[0] < self.min_face_size]
    batch_boxes = np.delete(batch_boxes, small_face_indices, axis=0)
    batch_probs = np.delete(batch_probs, small_face_indices, axis=0)
    if len(batch_points) > 0:
      batch_points = np.delete(batch_points, small_face_indices, axis=0)

# Extract faces
  faces = self.extract(img, batch_boxes, save_path)
  return batch_boxes, faces


### load model
resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(margin=44,image_size=182, keep_all=True, thresholds=[0.4, 0.5, 0.5], min_face_size=60)
mtcnn.detect_box = MethodType(detect_box, mtcnn)

# Initialize a list to store distances for dynamic thresholding
known_distances = []

### load known faces
saved_pictures = "/home/vishy/Documents/Attendance-Portal/C3_Members/"
all_people_faces = {}

# Define detection frequency
detection_counter = 0
detection_frequency = 3  # Adjust this value for your desired balance between accuracy and speed

# Iterate through folders containing individual's pictures
for person_folder in os.listdir(saved_pictures):
 person_path = os.path.join(saved_pictures, person_folder)
 for file in os.listdir(person_path):
  if file.endswith(".jpeg"):
   img = cv2.imread(os.path.join(person_path, file))
   cropped = mtcnn(img)
   if cropped is not None:
    all_people_faces[person_folder] = encode(cropped)[0, :]
    known_distances.append(encode(cropped)[0, :])

# Calculate a dynamic threshold based on the distances in the known dataset
known_distances = torch.stack(known_distances)
dynamic_threshold = np.percentile([dist.norm().item() for dist in known_distances], 90) # Adjust percentile as needed
     
     
# Initialize lists for storing predicted and true labels
predicted_labels = []
true_labels = []
confidence_scores = [] # List to store confidence scores

# Initialize a counter for correct predictions
correct_predictions = 0
total_predictions = 0

### initialize webcam and capture loop
cap = cv2.VideoCapture(0)
while True:
 # Capture frame-by-frame
 ret, frame = cap.read()
 frame = cv2.flip(frame, 1)
    
 # Skip detection every N frames (reduce processing)
 detection_counter += 1
 if detection_counter < detection_frequency:
    continue

 detection_counter = 0

 # Detect faces in the frame
 batch_boxes, cropped_images = mtcnn.detect_box(frame)

 # Recognition for each detected face
 if cropped_images is not None:
  for box, cropped in zip(batch_boxes, cropped_images):
   x, y, x2, y2 = [int(x) for x in box]
   img_embedding = encode(cropped.unsqueeze(0))

   min_distance = float("inf")
   min_key = "Undetected"
   for k, v in all_people_faces.items():
    distance = (v - img_embedding).norm().item()
    if distance < min_distance:
     min_distance = distance
     min_key = k
       
   total_predictions += 1

    # Use dynamic threshold
   if min_distance < dynamic_threshold:
      correct_predictions += 1  
       
   # Append predicted and true labels
   predicted_labels.append(min_key)
   true_labels.append(k) # Assuming file name is the actual label
   confidence_scores.append(1 - min_distance) # Confidence score (1 - distance)

   # Draw bounding box and label
   if min_distance < 0.7:
    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, min_key, (x + 5, y + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
   else:
    cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame, "Undetected", (x + 5, y + 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

 # Display the resulting frame
 cv2.imshow("Real-time face recognition", frame)

 # Quit loop on ' ' key press
 if cv2.waitKey(1) == ord(' '):
  break 

# Calculate accuracy
accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

# Calculate mean average precision (mAP)
mAP = average_precision_score((np.array(true_labels) != "Undetected").astype(int), confidence_scores)

# Print results
print(f"Accuracy: {accuracy * 100:.2f}% (Dynamic Threshold: {dynamic_threshold:.2f})")
print(f"Mean Average Precision (mAP): {mAP:.2f}")
   
# Release capture device
cap.release()
cv2.destroyAllWindows()