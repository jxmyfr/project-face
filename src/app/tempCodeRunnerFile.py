
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
            )
            
            # ใช้ landmark เพื่อ Align ใบหน้า
            aligned_face = align_face(frame, face_landmarks)
            aligned_face = aligned_face.astype('uint8')
            
            try:
                # ใช้ DeepFace.find เพื่อค้นหาความคล้ายในฐานข้อมูลที่มีอยู่ใน "my_faces"
                result = DeepFace.find(
                    img_path=aligned_face, 
                    db_path="D:/Study/Project/project-face/reference_images", 
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                if not result.empty:
                    name = result.iloc[0]["identity"]
                    cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            except Exception as e: