# recognition.py
import cv2
import threading
import queue
import time
import torch
import numpy as np
import pickle
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("ใช้ device:", device)

mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# โหลดฐานข้อมูลใบหน้าที่เทรนไว้
with open('known_faces.pkl', 'rb') as f:
    known_faces = pickle.load(f)

SIMILARITY_THRESHOLD = 0.75
frame_queue = queue.Queue(maxsize=5)

def capture_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            time.sleep(0.01)

def process_frames():
    tracker_list = []
    track_boxes = []
    detection_interval = 50  # ปรับให้ re-detect น้อยลง
    frame_count = 0
    detection_scale = 0.5  # ใช้ scale factor เพื่อลดความละเอียดสำหรับ detection
    while True:
        if frame_queue.empty():
            time.sleep(0.01)
            continue
        frame = frame_queue.get()
        frame_disp = frame.copy()
        frame_count += 1

        # สำหรับการ re-detect ใบหน้า (ทุก detection_interval เฟรม)
        if frame_count % detection_interval == 1:
            # ลดความละเอียดของภาพสำหรับการตรวจจับ
            small_frame = cv2.resize(frame, (0,0), fx=detection_scale, fy=detection_scale)
            small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            boxes_small, _ = mtcnn.detect(small_frame_rgb)
            tracker_list = []
            track_boxes = []
            if boxes_small is not None:
                # ขยาย bounding boxes กลับเป็นขนาดเดิม
                for box in boxes_small:
                    x1, y1, x2, y2 = [int(b/detection_scale) for b in box]
                    h, w, _ = frame.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                    track_boxes.append(bbox)
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, bbox)
                    tracker_list.append(tracker)
        else:
            new_boxes = []
            for tracker in tracker_list:
                success, bbox = tracker.update(frame)
                if success:
                    new_boxes.append(bbox)
                else:
                    new_boxes.append(None)
            track_boxes = new_boxes

        # ประมวลผลการจดจำใบหน้าสำหรับแต่ละ bounding box
        if track_boxes:
            for bbox in track_boxes:
                if bbox is None:
                    continue
                x, y, w_box, h_box = [int(v) for v in bbox]
                x2, y2 = x + w_box, y + h_box
                face_bgr = frame[y:y2, x:x2]
                if face_bgr.size == 0:
                    continue
                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                face_img = cv2.resize(face_rgb, (160, 160))
                face_tensor = torch.tensor(face_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                face_tensor = face_tensor.unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = resnet(face_tensor).cpu()
                name = "Unknown"
                similarities = {}
                for person, known_emb in known_faces.items():
                    sim = F.cosine_similarity(embedding, known_emb).item()
                    similarities[person] = sim
                if similarities:
                    best_match = max(similarities, key=similarities.get)
                    best_sim = similarities[best_match]
                    if best_sim > SIMILARITY_THRESHOLD:
                        name = best_match
                cv2.rectangle(frame_disp, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_disp, name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Real-time Face Recognition", frame_disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้")
        return
    capture_thread = threading.Thread(target=capture_frames, args=(cap,), daemon=True)
    capture_thread.start()

    try:
        process_frames()
    except KeyboardInterrupt:
        pass
    cap.release()

if __name__ == "__main__":
    main()
