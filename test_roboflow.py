import cv2
from inference_sdk import InferenceHTTPClient


API_KEY = "qUJ8iN6zQ495W6hemNLo"
MODEL_ID = "yolo-basket-fycvl/1"

VIDEO_PATH = r"C:\Users\miche\Desktop\Basket-AR\dataset\test\tiroDaDue1\clip_001827.mp4"

CONF_THRESHOLD = 0.20
FRAME_STEP = 1

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

CLASS_RENAME = {
    "Basket-AR - vdataset basket-ar": "Canestro",
    "------------------------------":"Palla"
}


def draw_predictions(frame, predictions):
    for pred in predictions:
        conf = pred["confidence"]

        if conf < CONF_THRESHOLD:
            continue

        original_class_name = pred["class"]
        class_name = CLASS_RENAME.get(original_class_name, original_class_name)

        x_center = pred["x"]
        y_center = pred["y"]
        width = pred["width"]
        height = pred["height"]

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        if class_name.lower() in ["palla", "ball"]:
            color = (0, 165, 255)  # arancione
        elif class_name.lower() in ["canestro", "rim", "hoop"]:
            color = (0, 255, 0)    # verde
        else:
            color = (255, 0, 0)    # blu

        label = f"{class_name} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    return frame


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Impossibile aprire il video: {VIDEO_PATH}")

frame_index = 0
last_predictions = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_index % FRAME_STEP == 0:
        result = client.infer(
            frame,
            model_id=MODEL_ID
        )

        last_predictions = result.get("predictions", [])

        print(f"\nFrame {frame_index}")
        for pred in last_predictions:
            print(
                pred["class"],
                round(pred["confidence"], 3),
                round(pred["x"], 1),
                round(pred["y"], 1),
                round(pred["width"], 1),
                round(pred["height"], 1)
            )

    frame_with_boxes = draw_predictions(frame.copy(), last_predictions)

    cv2.putText(
        frame_with_boxes,
        f"Frame: {frame_index}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("Roboflow - Palla e Canestro", frame_with_boxes)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()