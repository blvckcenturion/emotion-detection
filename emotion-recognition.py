import cv2
import mediapipe as mp
import math

# Video Capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Creating our drawing function
mp_drawing = mp.solutions.drawing_utils
drawing_settings = mp_drawing.DrawingSpec(thickness=1, circle_radius=2) # Drawing settings

# Creating an objext to store the face mesh
mp_face_mesh = mp.solutions.face_mesh # Calling the function
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1) # Creating the object

# Create the main while
while True:
    ret, frame = cap.read()
    # Color correction
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # Check the results
    results = face_mesh.process(frame_rgb)
    
    # Create a list where to store the results
    px = []
    py = []
    list = []
    r = 5
    t = 3
    # Check if there are results
    if results.multi_face_landmarks: # If there are results
        for face in results.multi_face_landmarks: # For each face
            mp_drawing.draw_landmarks(frame, face,mp_face_mesh.FACEMESH_CONTOURS, drawing_settings, drawing_settings) # Draw the landmarks
            # Get the points
            for id, points in enumerate(face.landmark):
                h, w, c = frame.shape
                x, y = int(points.x*w), int(points.y*h)
                px.append(x)
                py.append(y)
                list.append([id,x,y])
                if len(list) == 468:
                    # Right Eyebrow
                    x1,y1 = list[65][1:]
                    x2,y2 = list[158][1:]
                    cx,cy = (x1+x2) // 2, (y1+y2) // 2
                    longitud1 = math.hypot(x2-x1, y2-y1)
                    print(f"Right Eyebrow: {longitud1}")

                    # Left Eyebrow
                    x3, y3 = list[295][1:]
                    x4, y4 = list[358][1:]
                    cx2, cy2 = (x3+x4) // 2, (y3+y4) // 2
                    longitud2 = math.hypot(x4-x3, y4-y3)
                    print(f"Left Eyebrow: {longitud2}")

                    # Mouth 
                    x5, y5 = list[78][1:]
                    x6, y6 = list[308][1:]
                    cx3, cy3 = (x5+x6) // 2, (y5+y6) // 2
                    longitud3 = math.hypot(x6-x5, y6-y5)
                    print(f"Mouth: {longitud3}")

                    # Mouth Aperture
                    x7, y7 = list[13][1:]
                    x8, y8 = list[14][1:]
                    cx4, cy4 = (x7+x8) // 2, (y7+y8) // 2
                    longitud4 = math.hypot(x8-x7, y8-y7)
                    print(f"Mouth Aperture: {longitud4}")
                    

                    # Classifiction of emotions
                    if longitud1 < 29 and longitud3 > 80 and longitud3 < 105 and longitud4 < 10:
                        cv2.putText(frame, "Enojado", (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    elif longitud1 > 30 and longitud1 < 40 and longitud2 > 30 and longitud3 > 95 and longitud4 > 0 and longitud4 < 10:
                        cv2.putText(frame, "Feliz", (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                    elif longitud1 > 35 and longitud2 > 35 and longitud3 > 80 and longitud3 < 90 and longitud4 > 20:
                        cv2.putText(frame, "Asombrado", (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    elif longitud1 > 30 and longitud1 < 45 and longitud2 > 30 and longitud3 > 80 and longitud3 < 95 and longitud4 < 5:
                        cv2.putText(frame, "Triste", (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                    


    cv2.imshow('Video', frame)
    t = cv2.waitKey(1)
    if t == 27:
        break
    
cap.release()
cv2.destroyAllWindows()



