from tracker.bytrack import UltralyticsByteTrack
import cv2

tracker = UltralyticsByteTrack(model_path="yolov8m.pt")

cap = cv2.VideoCapture("data/sample_walking_video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    people = tracker.track_frame(frame)
    for person in people:
        x1, y1, x2, y2 = person["bbox"]
        track_id = person["track_id"]
        conf = person["confidence"]
        crop = person["crop"]  # ready for CLIP/MoonDream/embedding
        # Visualize
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("Track", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
