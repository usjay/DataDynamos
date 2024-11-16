import cv2
import os

# Path to the dataset and the directory to save cropped faces
dataset_path = r'C:\Users\ADMIN\Downloads\photos me'
cropped_faces_path = r'C:\Users\ADMIN\Downloads\photos me\cropped_faces'

# Create a directory to save cropped faces if it doesn't exist
if not os.path.exists(cropped_faces_path):
    os.makedirs(cropped_faces_path)

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to crop face from an image
def crop_face(image_path, save_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    if len(faces) == 0:
        print(f"No faces detected in {image_path}")
        return
    
    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        face_filename = os.path.join(save_path, os.path.basename(image_path))
        cv2.imwrite(face_filename, cropped_face)
        break  # Only crop the first detected face

# Loop through all images in the dataset and crop faces
for image_name in os.listdir(dataset_path):
    image_path = os.path.join(dataset_path, image_name)
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Ensure the file is an image
        crop_face(image_path, cropped_faces_path)
    else:
        print(f"Skipping non-image file {image_name}")

print("Face cropping completed.")
