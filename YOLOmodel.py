from ultralytics import YOLO

model= YOLO("yolo11x.pt")

risultato=model(source="C:/Users/miche/Desktop/Basket-AR/dataset/train/tiroLibero1/clip_003277.mp4",show=True, classes=[32],conf=0.05,imgsz=1280)