import cv2
from tracker.bytrack import UltralyticsByteTrack
from embedding.moondream_embedder import MoonDreamEmbedder
from PIL import Image
import os

# Set your input and output image paths
image_path = "data/friends3.png"
output_path = "output/desc_friends1_frame.jpg"

# Initialize models
tracker = UltralyticsByteTrack(model_path="yolov8m.pt")
moondream = MoonDreamEmbedder()

prompt = (
    "Describe only this person's visible physical characteristics and clothing in detail for re-identification. "
    "Use the following template and fill in as much detail as visible (write 'unknown' if you can't tell):\n\n"
    "- Estimated age range:\n"
    "- Gender:\n"
    "- Hair (color, length, style):\n"
    "- Skin (tone/complexion):\n"
    "- Face (shape, notable features, facial hair, glasses, freckles or moles, eyebrow shape):\n"
    "- Physical build (height estimate if possible, body type):\n"
    "- Clothing:\n"
    "    - Upper garment (type, color, pattern, fit):\n"
    "    - Lower garment (type, color, pattern, fit):\n"
    "    - Outerwear (type, color, pattern):\n"
    "    - Footwear (type, color):\n"
    "    - Accessories (hats, bags, watches, etc.):\n"
    "- Distinguishing marks (tattoos, scars, jewelry):\n\n"
    "Ignore background, scene, pose, activity, facial expression, other people, or any non-person objects.\n"
    "Return the information exactly in this template."
)

# Read image
frame = cv2.imread(image_path)
if frame is None:
    raise ValueError(f"Could not read image: {image_path}")

people = tracker.track_frame(frame)  # For one image, just detection + fake track ID

for person in people:
    x1, y1, x2, y2 = person["bbox"]
    track_id = person["track_id"] if "track_id" in person else None
    crop = person["crop"]
    if crop is not None and crop.shape[0] > 10 and crop.shape[1] > 10:
        try:
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            description = moondream.describe(
                pil_crop,
                prompt=prompt
            )
        except Exception as e:
            description = "[MoonDream error]"
            print(f"MoonDream error: {e}")

        person["description"] = description
        print(description)
        # Draw bounding box and description
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            frame,
            f"{'ID '+str(track_id)+':' if track_id else ''} {description[:60]}",
            (x1, max(y1-20, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0,255,0), 2, cv2.LINE_AA
        )

# Save or show the result
cv2.imshow("Detection+Description", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
