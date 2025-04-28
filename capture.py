import cv2
import os
import time
from facenet_pytorch import MTCNN
import torch

# ตั้งค่า device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

# โฟลเดอร์สำหรับเก็บรูปใบหน้า
reference_folder = 'reference_images'
os.makedirs(reference_folder, exist_ok=True)

def capture_faces(person_name):
    person_folder = os.path.join(reference_folder, person_name)
    os.makedirs(person_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    count = 0
    total_photos = 20
    initial_countdown = 3      # นับถอยหลังก่อนเริ่มถ่าย (วินาที)
    inter_capture_delay = 2    # นับถอยหลังระหว่างถ่ายแต่ละรูป (วินาที)
    
    print(f"เริ่มบันทึกใบหน้าของ {person_name}. กด 'q' ในหน้าต่างวีดีโอเพื่อออก หรือถ่ายครบ {total_photos} รูป")
    
    # Initial countdown: แสดงผลผ่านเทอร์มินัล
    for i in range(initial_countdown, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        captured_this_frame = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)
        
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                # วาด bounding box เท่านั้น (ไม่แสดงข้อความ)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                # บันทึกภาพใบหน้า
                img_path = os.path.join(person_folder, f"{person_name}_{int(time.time())}_{count}.jpg")
                cv2.imwrite(img_path, face_img)
                count += 1
                captured_this_frame = True
                if count >= total_photos:
                    break
        cv2.imshow("Capture Faces", frame)
        
        # หากถ่ายรูปในเฟรมนี้ ให้รอ inter_capture_delay วินาที โดยแสดงผลในเทอร์มินัล
        if captured_this_frame:
            for i in range(inter_capture_delay, 0, -1):
                print(f"Next capture in {i}...")
                time.sleep(1)
        
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= total_photos:
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"บันทึกใบหน้าของ {person_name} เรียบร้อยแล้ว ({count} รูป)")

if __name__ == "__main__":
    person_name = input("กรุณาใส่ชื่อบุคคล: ").strip()
    if person_name:
        capture_faces(person_name)
