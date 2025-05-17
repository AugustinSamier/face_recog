import tkinter as tk
import numpy as np
import os
import cv2
import face_recognition
from datetime import datetime

faces_images=[]
faces_labels=[]

faces_file=os.listdir("Faces")

for person in faces_file:
    person_path = os.path.join("Faces", person)
    if not os.path.isdir(person_path):
        continue
    try:
        faces=os.listdir(person_path)
        for face in faces:
            current_face=cv2.imread(f"Faces/{person}/{face}")
            if current_face is not None:
                faces_images.append(current_face)
                faces_labels.append(person)
            else:
                print(f"Image non lue: Faces/{person}/{face}")
    except:
        print(f"Erreur avec Faces/{person}/{face}")
        pass

def get_face_encodings(images):
    encoding_list=[]
    for image in images:
        if image is None or image.dtype != np.uint8:
            print("[SKIP] Image invalide pour face_encodings()")
            continue
        try:
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            encodings=face_recognition.face_encodings(image)
        except:
            print(image.type())
        if encodings:
            encoding_list.append(encodings)
    return encoding_list

def document_recognised_face(name,filename="records.csv"):
    capture_date=datetime.now().strftime("%Y-%m-%d")
    if not os.path.isfile(filename):
        with open(filename,"w") as f:
            f.write("Name,Date,Time")
        
    with open(filename,"r+") as file:
        lines=file.readlines()
        existing_names=[line.split(",")[0] for line in lines]
        if name not in existing_names:
            now=datetime.now()
            current_time=now.strftime("%H:%M:%S")
            file.write(f"\n{name},{capture_date},{current_time}")

known_face_encodings=get_face_encodings(faces_images)

def start_recognition_program():
    for i in range(3):
        video_capture = cv2.VideoCapture(i)
        if video_capture.read()[0]:
            print(f"Webcam détectée sur l'index {i}")
            break
    while True:
        frame=video_capture.read()
        if frame is not None:
            frame=frame[1]
            resized_frame=cv2.resize(frame,(0,0),None,0.25,0.25)
            resized_frame=cv2.cvtColor(resized_frame,cv2.COLOR_BGR2RGB)

            face_locations=face_recognition.face_locations(resized_frame)
            current_face_encodings=face_recognition.face_encodings(resized_frame,face_locations)
            for face_encoding, location in zip(current_face_encodings,face_locations):
                matches=face_recognition.compare_faces(known_face_encodings,face_encoding)
                face_distances=face_recognition.face_distance(known_face_encodings,face_encoding)
                best_match_index=np.argmin(face_distances)
                if matches[best_match_index]:
                    recognized_name=faces_labels[best_match_index].upper()
                    top,right,bottom,left=location
                    top,right,bottom,left=top*4,right*4,bottom*4,left*4
                    cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
                    cv2.rectangle(frame,(left,bottom-35),(right,bottom),(0.255,0),cv2.FILLED)
                    cv2.putText(frame,recognized_name,(left+6,bottom-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                    document_recognised_face(recognized_name)
            cv2.imshow("Webcam",frame)
        key=cv2.waitKey(1) & 0xFF
        if key==ord("q"):
            break
    video_capture.release()
    cv2.destroyAllWindows()

root=tk.Tk()
root.title("Face Recognition Program")

label=tk.Label(root,text="Click the button to start the facial recognition program")
label.pack(pady=10)

start_button=tk.Button(root,text="Start Recognition",command=start_recognition_program)
start_button.pack(pady=10)

def quit_app():
    root.quit()
    cv2.destroyAllWindows()

exit_button=tk.Button(root,text="Close",command=quit_app)
exit_button.pack(pady=10)

root.mainloop()