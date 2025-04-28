from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
import os
import numpy as np
from PIL import Image

# ตั้งค่า device (GPU หากมี, ถ้าไม่มีใช้ CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# สร้าง detector (MTCNN) และ recognition model (InceptionResnetV1)
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ฟังก์ชันโหลด reference images และคำนวณ embedding
def load_reference_images(folder):
    names = []
    embeddings = []
    for file in os.listdir(folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, file)
            img = Image.open(img_path)
            face = mtcnn(img)
            if face is not None:
                # face shape: [3, 160, 160]
                face = face.unsqueeze(0).to(device)  # shape [1, 3, 160, 160]
                embedding = resnet(face)  # shape [1, 512]
                embeddings.append(embedding.detach().cpu())
                names.append(os.path.splitext(file)[0])
    if embeddings:
        embeddings = torch.cat(embeddings)  # shape [N, 512]
    else:
        embeddings = None
    return names, embeddings

# กำหนด path ของฐานข้อมูลใบหน้า (reference images)
ref_folder = "reference_images"  # ให้สร้างโฟลเดอร์นี้ไว้ในตำแหน่งที่โค้ดเข้าถึงได้
ref_names, ref_embeddings = load_reference_images(ref_folder)
print("Loaded reference embeddings for:", ref_names)

# เริ่มต้นกล้อง
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # แปลงภาพเป็น RGB และแปลงเป็น PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # ตรวจจับใบหน้าด้วย MTCNN
    face = mtcnn(img)
    if face is not None:
        # ทำการคำนวณ embedding
        face = face.unsqueeze(0).to(device)
        embedding = resnet(face)  # shape [1, 512]
        
        if ref_embeddings is not None:
            # คำนวณ cosine similarity ระหว่าง embedding ของ face กับ reference embeddings
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            sims = cos(embedding, ref_embeddings)
            max_sim, idx = torch.max(sims, dim=0)
            
            # กำหนด threshold (ปรับได้ตามต้องการ)
            threshold = 0.8
            if max_sim > threshold:
                name = ref_names[idx]
                cv2.putText(frame, f"{name} ({max_sim:.2f})", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    cv2.imshow("Modern Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
