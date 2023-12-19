import os
import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm
from types import MethodType

from sklearn.metrics import accuracy_score

### helper function
def encode(img):
  res = resnet(torch.Tensor(img))
  return res

def detect_box(self, img, save_path=None):
  batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
  
  # Select faces
  if not self.keep_all:
    batch_boxes, batch_probs, batch_points = self.select_boxes(
        batch_boxes, batch_probs, batch_points, img, method=self.selection_method
    )
  # Extract faces
  faces = self.extract(img, batch_boxes, save_path)
  return batch_boxes, faces


### load model
resnet = torch.load("Attendify_model.pth").eval()
mtcnn = MTCNN(margin=44,image_size=182, keep_all=True, thresholds=[0.6, 0.6, 0.6], min_face_size=60)
mtcnn.detect_box = MethodType(detect_box, mtcnn)

### load known faces
saved_pictures = "/home/vishy/Projects/Attendance/C3_Members/"
all_people_faces = {}

# Iterate through folders containing individual's pictures
for person_folder in os.listdir(saved_pictures):
  person_path = os.path.join(saved_pictures, person_folder)
  for file in os.listdir(person_path):
    if file.endswith(".jpeg") or file.endswith(".jpg"):
      img = cv2.imread(os.path.join(person_path, file))
      cropped = mtcnn(img)
      if cropped is not None:
        all_people_faces[person_folder] = encode(cropped)[0, :]

### initialize webcam and capture loop
cap = cv2.VideoCapture(0)
while True:
  # Capture frame-by-frame
  ret, frame = cap.read()
  frame = cv2.flip(frame, 1)

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
      

      # Calculate accuracy for this recognized face
      if min_distance < 0.7:
        recognized_accuracy = (1 - min_distance) * 100 + 45
      else:
        recognized_accuracy = 0

      # Update the box drawing logic to include accuracy
      accuracy_text = f"{recognized_accuracy:.2f}%"
      cv2.putText(frame, accuracy_text, (x2 + 5, y2 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)


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


# Release capture device
cap.release()
cv2.destroyAllWindows()