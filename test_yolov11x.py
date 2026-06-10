import cv2
from ultralytics import YOLO


# ==========================
# MODIFICA QUESTI PERCORSI
# ==========================

BEST_MODEL = "runs/detect/runs_basket/yolo11x_palla_canestro_dual_gpu_1280/weights/best.pt"

VIDEO_PATH = r"percorso/del/tuo/video.mp4"


# ==========================
# PARAMETRI INFERENZA
# ==========================

IMG_SIZE = 1280
CONF_THRESHOLD = 0.05
IOU_THRESHOLD = 0.50

DEVICE = 0  # usa GPU 0


# ==========================
# CARICAMENTO MODELLO
# ==========================

model = YOLO(BEST_MODEL)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Impossibile aprire il video: {VIDEO_PATH}")

frame_index = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Inferenza sul frame originale
    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=DEVICE,
        verbose=False
    )

    result = results[0]

    # Disegna le bounding box sul frame
    annotated_frame = result.plot()

    # Scrive numero frame
    cv2.putText(
        annotated_frame,
        f"Frame: {frame_index}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("YOLO11x - Palla e Canestro", annotated_frame)

    # Premi Q per uscire
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()