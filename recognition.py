import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import pickle
from concurrent.futures import ThreadPoolExecutor

# โหลดฐานข้อมูลใบหน้าที่เทรนไว้ (known_faces.pkl)
with open('known_faces.pkl', 'rb') as f:
    known_faces = pickle.load(f)

MATCH_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# ตั้งค่า MediaPipe Face Mesh แทน Face Detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# สำหรับวาด Face Mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def recognize_face(face_image):
    """
    รับ face_image (RGB) ที่ครอปแล้ว แล้วคำนวณ face encoding
    โดยระบุ known_face_locations เป็น bounding box ครอบคลุมทั้งภาพ
    คืนชื่อบุคคลที่ตรงกัน หรือ "Unknown"
    """
    face_image = np.ascontiguousarray(face_image)
    h, w, _ = face_image.shape
    face_locations = [(0, w, h, 0)]
    encodings = face_recognition.face_encodings(face_image, known_face_locations=face_locations)
    if not encodings:
        return "Unknown"
    face_encoding = encodings[0]

    names = list(known_faces.keys())
    known_encodings = np.array(list(known_faces.values()))
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_match_index = np.argmin(distances)
    if distances[best_match_index] < MATCH_THRESHOLD:
        return names[best_match_index]
    else:
        return "Unknown"

def bbox_similarity(bbox1, bbox2):
    """
    คำนวณ IoU ระหว่าง bounding box สองอัน
    bbox format: (x, y, w, h)
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1+w1, x2+w2)
    yi2 = min(y1+h1, y2+h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้")
        return

    prev_faces = []
    executor = ThreadPoolExecutor(max_workers=2)

    mode = 2  # เริ่มต้นด้วย mode 2 (bounding box)
    print("[INFO] กด 'q' เพื่อออก, กด '1' เพื่อแสดง Face Mesh เต็ม, กด '2' เพื่อแสดง bounding box")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            mode = 1
        elif key == ord('2'):
            mode = 2
        elif key == ord('q'):
            break

        frame_disp = frame.copy()
        ih, iw, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ใช้ face_mesh.process() แทน face_detection.process()
        mesh_results = face_mesh.process(rgb_frame)
        current_faces = []

        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                # คำนวณ bounding box จาก min/max ของแลนด์มาร์กทั้งหมด
                xs = []
                ys = []
                for lm in face_landmarks.landmark:
                    xs.append(int(lm.x * iw))
                    ys.append(int(lm.y * ih))

                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                # ปรับขอบเขตไม่ให้ออกนอกภาพ
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(iw, x_max)
                y_max = min(ih, y_max)

                w_box = x_max - x_min
                h_box = y_max - y_min
                bbox = (x_min, y_min, w_box, h_box)

                # ครอปรูปใบหน้า
                face_roi = rgb_frame[y_min:y_min+h_box, x_min:x_min+w_box]
                if face_roi.size == 0:
                    continue

                # ตรวจ cache
                cached_name = None
                cached_future = None
                for prev_bbox, prev_name, prev_future in prev_faces:
                    if bbox_similarity(bbox, prev_bbox) > IOU_THRESHOLD:
                        if prev_name != "Unknown":
                            cached_name = prev_name
                            cached_future = prev_future
                        break

                if cached_name is not None:
                    if cached_future is not None and cached_future.done():
                        recognized_name = cached_future.result()
                    else:
                        recognized_name = cached_name
                else:
                    # ส่งงาน deep recognition
                    future = executor.submit(recognize_face, face_roi)
                    recognized_name = "Processing"
                    cached_future = future

                current_faces.append((bbox, recognized_name, cached_future))

                # แสดงผลตามโหมด
                if mode == 2:
                    # วาด bounding box + ชื่อ
                    cv2.rectangle(frame_disp, (x_min, y_min), (x_min+w_box, y_min+h_box), (0, 255, 0), 2)
                    cv2.putText(frame_disp, recognized_name, (x_min, y_min-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                elif mode == 1:
                    # วาด Face Mesh เต็ม (tesselation)
                    mp_drawing.draw_landmarks(
                        image=frame_disp,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    # แสดงชื่อที่มุมบนซ้ายของ bounding box
                    cv2.putText(frame_disp, recognized_name, (x_min, y_min-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        prev_faces = current_faces

        cv2.imshow("Real-time Face Recognition with MediaPipe Face Mesh", frame_disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown()

if __name__ == "__main__":
    main()
