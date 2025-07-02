#tests/test_bytetrack_moondream.py
import cv2
from tracker.bytrack import UltralyticsByteTrack
from embedding.moondream_embedder import MoonDreamEmbedder
from PIL import Image

video_path = "data/friends1.mov"
output_path = "output/tracked_with_description.mp4"

tracker = UltralyticsByteTrack(model_path="yolov8m.pt")
moondream = MoonDreamEmbedder()

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    print(f"Processing frame: {frame_idx}")  # Add this line for debugging

    frame_idx += 1

    people = tracker.track_frame(frame)

    for person in people:
        x1, y1, x2, y2 = person["bbox"]
        track_id = person["track_id"]
        crop = person["crop"]
        if crop is not None and crop.shape[0] > 10 and crop.shape[1] > 10:  # Avoid tiny/invalid crops
            try:
                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                description = moondream.describe(
                    pil_crop,
                    prompt="Describe this person's appearance for re-identification."
                )
            except Exception as e:
                description = "[MoonDream error]"
                print(f"MoonDream error: {e}")

            person["description"] = description
            # Draw bounding box and description
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(
                frame,
                f"ID {track_id}: {description[:60]}",
                (x1, max(y1-20, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0,255,0), 2, cv2.LINE_AA
            )

    cv2.imshow("Tracking+Descriptions", frame)
    out_writer.write(frame)
    #if cv2.waitKey(1) & 0xFF == ord("q"):
        #break

#cap.release()
#out_writer.release()
#cv2.destroyAllWindows()
