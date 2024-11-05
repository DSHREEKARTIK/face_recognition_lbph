import cv2
import os

def collect_data_from_camera(output_dir):
    video_index = 0  # Default camera index
    video = cv2.VideoCapture(video_index)
    if not video.isOpened():
        print(f"Error: Could not open video capture with index {video_index}")
        return

    facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    user_id = input("Enter Your ID: ")
    id_dir = os.path.join(output_dir, str(user_id))
    if not os.path.exists(id_dir):
        os.makedirs(id_dir)

    count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Could not read frame from video capture")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(os.path.join(id_dir, f"User.{user_id}.{count}.jpg"), gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord('q') or count >= 500:  # Press 'q' to exit
            break

    video.release()
    cv2.destroyAllWindows()
    print("Dataset Collection Done.")

def collect_data_from_gallery(output_dir, gallery_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(gallery_dir):
        for file in files:
            if file.endswith(".jpg"):
                user_id = os.path.basename(root)
                id_dir = os.path.join(output_dir, user_id)
                if not os.path.exists(id_dir):
                    os.makedirs(id_dir)
                image_path = os.path.join(root, file)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(os.path.join(id_dir, file), img)

    print("Dataset Collection from Gallery Done.")

def main():
    output_dir = "datasets"
    choice = input("Choose data collection method:\n1. Live Camera\n2. Gallery\nEnter your choice: ")

    if choice == "1":
        collect_data_from_camera(output_dir)
    elif choice == "2":
        gallery_dir = input("Enter path to the gallery directory: ")
        collect_data_from_gallery(output_dir, gallery_dir)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
