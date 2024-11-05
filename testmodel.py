import cv2

def load_models():
    facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Trainer.yml")
    return facedetect, recognizer

def recognize_faces(facedetect, recognizer, names, confidence_threshold=70):
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open video capture")
        return

    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Could not read frame from video capture")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            label, conf = recognizer.predict(roi_gray)
            
            # Debugging: Print label and confidence
            print(f"Detected label: {label}, Confidence: {conf}")

            if conf < confidence_threshold:
                name = names.get(label, "Unknown")
            else:
                name = "Unknown"

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    facedetect, recognizer = load_models()
    names = {1: "karthik", 2: "rahul", 3: "dinesh", 4: "kavya", 5: "uday", 6: "Geethika", 7: "Bharani"}
    recognize_faces(facedetect, recognizer, names)
