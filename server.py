import cv2
import time
import mediapipe as mp
import face_recognition
import numpy as np
from flask import Flask, Response

app = Flask(__name__)

# ตั้งค่า MediaPipe Face Mesh หรือ Face Detection (ตามต้องการ)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# โหลดฐานข้อมูลใบหน้าที่เทรนไว้
import pickle
with open('known_faces.pkl', 'rb') as f:
    known_faces = pickle.load(f)

MATCH_THRESHOLD = 0.5

def recognize_face(face_image):
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

def gen_frames():
    cap = cv2.VideoCapture(4)
    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้")
        return
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # แปลงเป็น RGB สำหรับ MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                xs = []
                ys = []
                for lm in face_landmarks.landmark:
                    xs.append(int(lm.x * frame.shape[1]))
                    ys.append(int(lm.y * frame.shape[0]))
                
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                # ครอปใบหน้า
                face_roi = rgb_frame[y_min:y_max, x_min:x_max]
                if face_roi.size != 0:
                    name = recognize_face(face_roi)
                else:
                    name = "Unknown"

                # วาดกรอบและชื่อ
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
                cv2.putText(frame, name, (x_min, y_min-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # แปลงเป็น JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # ส่งเป็น multipart/x-mixed-replace (MJPEG)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "Video streaming route: /video_feed"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
