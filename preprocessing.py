import os
import cv2
from PIL import Image
import numpy as np

face_cascade = cv2.CascadeClassifier("C:/Users/ass85/PycharmProjects/face_recognition_project/.venv/Scripts/haarcascade_frontalface_default .xml")

def extract_face(img, output_size=(160, 160)):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_roi_gray = img_gray[y:y + h, x:x + w]
        face_roi_resized = cv2.resize(face_roi_gray, output_size, interpolation=cv2.INTER_AREA)
        return face_roi_resized

    return None

def load_faces(img_path, output_size=(160, 160)):
    img = cv2.imread(img_path)
    face_detected = extract_face(img, output_size)

    if face_detected is not None and face_detected.any():
        print(face_detected)
        return face_detected

def load_dataset(parent_dir, output_size=(160, 160)):
    images = []
    labels = []
    i=1
    for child_dir in os.listdir(parent_dir):
        child_path = os.path.join(parent_dir, child_dir)
        if os.path.isdir(child_path):
            print(f"Processing images in {child_dir}")

            for img_file in os.listdir(child_path):
                img_path = os.path.join(child_path, img_file)

                # Vérifier si le fichier est une image (vous pouvez ajouter plus d'extensions si nécessaire)
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image = load_faces(img_path, output_size)

                    if image is not None:
                        images.append(image)
                        labels.append(child_dir)
                        if child_dir == "Mbappe":
                            # Convertir l'image NumPy en image PIL
                            pil_image = Image.fromarray(image)

                            # Sauvegarder l'image au format PNG
                            pil_image.save(f"image{i}.png")
                            i += 1

    images_array = np.array(images)
    np.savez_compressed("images_dataset.npz", images=images_array, labels=labels)

load_dataset("C:/Users/ass85/PycharmProjects/face_recognition_project/.venv/Scripts/images_for_face_recognition")
