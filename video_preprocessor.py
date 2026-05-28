import cv2
import numpy as np


class VideoPreprocessor:
    def __init__(self, max_frame, img_size=224):
        self.max_frame = max_frame
        self.img_size = img_size

    def estrai_frame_da_video(self, percorso_video):
        cap = cv2.VideoCapture(percorso_video)

        if not cap.isOpened():
            raise ValueError(f"Errore apertura video: {percorso_video}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        mask = []

        if total_frames >= self.max_frame:
            #PERFORMANCE
            indici = np.linspace(
                0,
                total_frames - 1,
                self.max_frame,
                dtype=int
            )

            indici = set(indici.tolist())

            current_index = 0

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if current_index in indici:
                    frame = cv2.resize(frame, (self.img_size, self.img_size))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    frames.append(frame)
                    mask.append(1)

                current_index += 1

                if len(frames) == self.max_frame:
                    break

        else:

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frames.append(frame)
                mask.append(1)

        cap.release()

        while len(frames) < self.max_frame:
            zero_frame = np.zeros(
                (self.img_size, self.img_size, 3),
                dtype=np.uint8
            )

            frames.append(zero_frame)
            mask.append(0)

        frames = np.array(frames, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)

        return frames, mask, total_frames

    def __call__(self, percorso_video):
        return self.estrai_frame_da_video(percorso_video)