# train.py
import os
import cv2
import torch
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

reference_folder = 'reference_images'
known_faces = {}

def load_known_faces():
    for person_name in os.listdir(reference_folder):
        person_path = os.path.join(reference_folder, person_name)
        if not os.path.isdir(person_path):
            continue
        embeddings = []
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = mtcnn(img_rgb)
            if face is None:
                continue
            if face.dim() == 3:
                face = face.unsqueeze(0)
            with torch.no_grad():
                emb = resnet(face.to(device))
            embeddings.append(emb.cpu())
        if embeddings:
            mean_embedding = torch.cat(embeddings).mean(0, keepdim=True)
            known_faces[person_name] = mean_embedding
            print(f"โหลด embedding ของ {person_name} เรียบร้อย")
    # บันทึก known_faces ลงในไฟล์
    with open('known_faces.pkl', 'wb') as f:
        pickle.dump(known_faces, f)
    print("การเทรนใบหน้าสำเร็จแล้ว บันทึกลง known_faces.pkl")

if __name__ == "__main__":
    load_known_faces()
