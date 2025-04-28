import os
import cv2
import face_recognition
import numpy as np
import pickle

# โฟลเดอร์ที่เก็บรูปใบหน้าของแต่ละบุคคล
reference_folder = "reference_images"
known_faces = {}

# สแกนแต่ละโฟลเดอร์ใน reference_images
for person_name in os.listdir(reference_folder):
    person_folder = os.path.join(reference_folder, person_name)
    if not os.path.isdir(person_folder):
        continue
    
    encodings = []
    # อ่านไฟล์ภาพในโฟลเดอร์ของบุคคลนั้น
    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # แปลงภาพจาก BGR เป็น RGB สำหรับ face_recognition
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # คำนวณ face encoding (จะได้เป็น list ถ้ามีใบหน้ามากกว่า 1 แต่ปกติจะมีแค่ใบหน้าเดียว)
        face_encs = face_recognition.face_encodings(rgb_image)
        if face_encs:
            encodings.append(face_encs[0])
    
    # หากมีการคำนวณ encoding ได้ ให้คำนวณค่าเฉลี่ยเพื่อเป็นตัวแทนของบุคคลนั้น
    if encodings:
        mean_encoding = np.mean(encodings, axis=0)
        known_faces[person_name] = mean_encoding
        print(f"เทรนใบหน้าของ {person_name} เรียบร้อย (จำนวนภาพ: {len(encodings)})")

# บันทึกฐานข้อมูลใบหน้าลงในไฟล์ known_faces.pkl
with open("known_faces.pkl", "wb") as f:
    pickle.dump(known_faces, f)

print("บันทึกฐานข้อมูลใบหน้าลงใน known_faces.pkl เรียบร้อย")
