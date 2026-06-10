import cv2
from pathlib import Path
from rfdetr import RFDETRLarge


# ============================================================
# MODIFICA SOLO QUESTI PERCORSI
# ============================================================

CHECKPOINT_PATH = r"C:\Users\miche\Desktop\Basket-AR\dataset\checkpoint_best_total.pth"

VIDEO_PATH = r"C:\Users\miche\Desktop\Basket-AR\dataset\test\idle\clip_001722.mp4"


# ============================================================
# PARAMETRI
# ============================================================

CONF_THRESHOLD = 0.40

# Se hai addestrato RF-DETR Large a 704, lascia 704
INPUT_SIZE = 704

CLASS_NAMES = {
    0: "Palla",
    1: "Canestro"
}


# ============================================================
# DISEGNO BOUNDING BOX
# ============================================================

def draw_detections(frame, detections):
    if detections is None:
        return frame

    if len(detections) == 0:
        return frame

    for xyxy, class_id, conf in zip(
        detections.xyxy,
        detections.class_id,
        detections.confidence
    ):
        if conf < CONF_THRESHOLD:
            continue

        class_id = int(class_id)
        class_name = CLASS_NAMES.get(class_id, str(class_id))

        x1, y1, x2, y2 = map(int, xyxy)

        if class_name == "Palla":
            color = (0, 165, 255)  # arancione
        else:
            color = (0, 255, 0)    # verde

        label = f"{class_name} {conf:.2f}"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

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


def main():
    checkpoint_file = Path(CHECKPOINT_PATH)
    video_file = Path(VIDEO_PATH)

    print("Checkpoint:", checkpoint_file)
    print("Video:", video_file)

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_file}")

    if not video_file.exists():
        raise FileNotFoundError(f"Video non trovato: {video_file}")

    print("\nCaricamento modello RF-DETR...")
    model = RFDETRLarge(
        pretrain_weights=str(checkpoint_file),
        num_classes=2
    )

    model.optimize_for_inference()

    cap = cv2.VideoCapture(str(video_file))

    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_file}")

    frame_index = 0

    cv2.namedWindow("RF-DETR PTH - Palla e Canestro", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # OpenCV legge in BGR, RF-DETR vuole RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = model.predict(
            frame_rgb,
            threshold=CONF_THRESHOLD,
            shape=(INPUT_SIZE, INPUT_SIZE),
            include_source_image=False
        )

        frame_with_boxes = draw_detections(frame.copy(), detections)

        cv2.putText(
            frame_with_boxes,
            f"Frame: {frame_index}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.imshow("RF-DETR PTH - Palla e Canestro", frame_with_boxes)

        # Premi Q per uscire
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\nTest completato.")


if __name__ == "__main__":
    main()