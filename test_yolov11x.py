import cv2
from ultralytics import YOLO


BEST_MODEL = r"C:\Users\miche\Desktop\Basket-AR\dataset\best.pt"
VIDEO_PATH = r"C:\Users\miche\Desktop\Basket-AR\dataset\test\tiroDaDue1\clip_001827.mp4"

IMG_SIZE = 1280
DEVICE = "cpu"

CLASS_NAMES = {
    0: "Palla",
    1: "Canestro"
}

CONF_THRESHOLDS = {
    0: 0.50,  
    1: 0.50   
}


def draw_boxes(frame, result):
    h_frame, w_frame = frame.shape[:2]

    if result.boxes is None:
        return frame

    for box in result.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])

        if class_id not in CLASS_NAMES:
            continue

        if conf < CONF_THRESHOLDS[class_id]:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        box_w = x2 - x1
        box_h = y2 - y1

        # elimina box assurde tipo linee enormi
        if box_w > w_frame * 0.50 and box_h < h_frame * 0.08:
            continue

        class_name = CLASS_NAMES[class_id]

        if class_name == "Palla":
            color = (0, 165, 255)  # arancione
        else:
            color = (0, 255, 0)    # verde

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name} {conf:.2f}"

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


model = YOLO(BEST_MODEL)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Impossibile aprire il video: {VIDEO_PATH}")

frame_index = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=0.03,
        iou=0.50,
        device=DEVICE,
        verbose=False
    )

    result = results[0]

    frame_with_boxes = draw_boxes(frame.copy(), result)

    cv2.putText(
        frame_with_boxes,
        f"Frame: {frame_index}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("YOLO - Palla e Canestro", frame_with_boxes)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()