import cv2
import numpy as np


class VideoPreprocessor:
    def __init__(self, max_frame, img_size):
        self.max_frame = max_frame
        self.img_size = img_size

    def crea_indici_inizio_fine(self, total_frames):
        """
        Campionamento uniforme lungo tutto il video per non perdere la parte centrale.
        """
        if total_frames <= self.max_frame:
            return list(range(total_frames))

        # Genera 'max_frame' indici distribuiti uniformemente dall'inizio alla fine
        indices = np.linspace(0, total_frames - 1, self.max_frame, dtype=int).tolist()
        return indices

    def estrai_frame_da_video(self, percorso_video):
        cap = cv2.VideoCapture(percorso_video)

        if not cap.isOpened():
            raise ValueError(f"Errore apertura video: {percorso_video}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        mask = []

        indici = self.crea_indici_inizio_fine(total_frames)
        indici = set(indici)

        current_index = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if current_index in indici:
                frame = cv2.resize(
                    frame,
                    (self.img_size, self.img_size)
                )

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                frames.append(frame)
                mask.append(1)

            current_index += 1

            if len(frames) == min(total_frames, self.max_frame):
                # Se il video è lungo, basta arrivare a max_frame.
                # Se il video è corto, basta arrivare a total_frames.
                break

        cap.release()

        # Padding finale con frame neri
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