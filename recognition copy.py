import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import pickle
import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor

# สำหรับ Windows: pip install pygrabber
try:
    from pygrabber.dshow_graph import FilterGraph
    _USE_DSHOW = True
except ImportError:
    _USE_DSHOW = False

# โหลดฐานข้อมูลใบหน้า
with open('known_faces.pkl', 'rb') as f:
    known_faces = pickle.load(f)

MATCH_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# ตั้งค่า MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=5, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def recognize_face(face_image):
    face_image = np.ascontiguousarray(face_image)
    h, w, _ = face_image.shape
    loc = [(0, w, h, 0)]
    encs = face_recognition.face_encodings(face_image, known_face_locations=loc)
    if not encs:
        return "Unknown"
    fe = encs[0]
    names = list(known_faces.keys())
    kek = np.array(list(known_faces.values()))
    dists = face_recognition.face_distance(kek, fe)
    idx = np.argmin(dists)
    return names[idx] if dists[idx] < MATCH_THRESHOLD else "Unknown"

def bbox_iou(b1, b2):
    x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
    xi1, yi1 = max(x1,x2), max(y1,y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    union = w1*h1 + w2*h2 - inter
    return inter/union if union else 0

def list_cameras(max_test=8):
    """
    คืน list ของ (index, ชื่อกล้อง) ตามแพลตฟอร์ม
    """
    cams = []
    system = platform.system()

    # Windows: DirectShow
    if system == "Windows" and _USE_DSHOW:
        graph = FilterGraph()
        names = graph.get_input_devices()
    else:
        names = []

    # Linux: v4l2-ctl
    if system == "Linux":
        try:
            p = subprocess.run(
                ["v4l2-ctl", "--list-devices"],
                capture_output=True, text=True, check=True
            )
            lines = p.stdout.splitlines()
            # วน parse แบบง่าย: ชื่อ device line นึง ตามด้วย /dev/video* บรรทัดถัดไป
            dev_map = {}
            last_name = None
            for ln in lines:
                if ln.strip() == "":
                    last_name = None
                    continue
                if not ln.startswith("\t"):  # ชื่อ device
                    last_name = ln.strip().rstrip(":")
                else:
                    dev = ln.strip().split()[0]
                    if last_name:
                        dev_map[dev] = last_name
            names = [dev_map.get(f"/dev/video{i}", "") for i in range(max_test)]
        except Exception:
            names = []

    # ทดสอบเปิดแต่ละ index
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            if system == "Linux":
                dev = f"/dev/video{i}"
                name = names[i] if i < len(names) and names[i] else dev
            else:
                name = names[i] if i < len(names) and names[i] else f"Camera {i}"
            cams.append((i, name))
            cap.release()

    return cams

def select_camera():
    cams = list_cameras()
    if not cams:
        print("ไม่พบกล้องบนเครื่อง")
        return 0
    print("Available cameras:")
    for idx, name in cams:
        print(f"  [{idx}] {name}")
    choice = input(f"Enter camera index [{cams[0][0]}]: ")
    try:
        ci = int(choice)
    except:
        ci = cams[0][0]
    return ci

def main():
    executor = ThreadPoolExecutor(max_workers=2)
    mode = 2  # 1=FaceMesh, 2=BBox

    cam_idx = select_camera()
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"ไม่สามารถเปิดกล้องหมายเลข {cam_idx} ได้")
        return

    prev = []
    print("[INFO] 'q' ออก | '1' Face Mesh | '2' BBox | 'c' เปลี่ยนกล้อง")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        disp = frame.copy()
        ih, iw = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        cur = []

        if res.multi_face_landmarks:
            for lm in res.multi_face_landmarks:
                xs = [int(p.x*iw) for p in lm.landmark]
                ys = [int(p.y*ih) for p in lm.landmark]
                x1, x2 = max(0,min(xs)), min(iw,max(xs))
                y1, y2 = max(0,min(ys)), min(ih,max(ys))
                w, h = x2-x1, y2-y1
                bbox = (x1,y1,w,h)
                roi = rgb[y1:y1+h, x1:x1+w]
                if roi.size==0: continue

                # cache lookup
                name, fut = None, None
                for pb,pn,pf in prev:
                    if bbox_iou(bbox, pb) > IOU_THRESHOLD:
                        name, fut = pn, pf
                        break

                if fut:
                    recognized = fut.result() if fut.done() else (name or "Processing")
                else:
                    fut = executor.submit(recognize_face, roi)
                    recognized = "Processing"

                cur.append((bbox, recognized, fut))

                # วาดผล
                if mode==2:
                    cv2.rectangle(disp, (x1,y1), (x1+w,y1+h), (0,255,0),2)
                    cv2.putText(disp, recognized, (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
                else:
                    mp_drawing.draw_landmarks(
                        image=disp, landmark_list=lm,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    cv2.putText(disp, recognized, (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

        prev = cur
        cv2.imshow("Face Recognition", disp)
        key = cv2.waitKey(1) & 0xFF

        if key==ord('q'):
            break
        elif key==ord('1'):
            mode=1
        elif key==ord('2'):
            mode=2
        elif key==ord('c'):
            cap.release()
            cam_idx = select_camera()
            cap = cv2.VideoCapture(cam_idx)
            if not cap.isOpened():
                print(f"ไม่สามารถเปิดกล้องหมายเลข {cam_idx} ได้")
                break

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown()

if __name__ == "__main__":
    main()
