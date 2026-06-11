import cv2
import numpy as np


class VideoPreprocessor:
    def __init__(self, max_frame, img_size):
        self.max_frame = max_frame
        self.img_size = img_size

    def crea_indici_uniformi(self, total_frames):
        """
        Se il video ha più di max_frame frame:
        seleziona max_frame indici uniformemente distribuiti
        dall'inizio alla fine del video.

        Se il video ha meno di max_frame frame:
        usa tutti i frame reali.
        """

        if total_frames <= 0:
            return []

        if total_frames <= self.max_frame:
            return list(range(total_frames))

        return np.linspace(
            0,
            total_frames - 1,
            self.max_frame,
            dtype=np.int64
        ).tolist()

    def estrai_frame_da_video(self, percorso_video):
        cap = cv2.VideoCapture(percorso_video)

        if not cap.isOpened():
            raise ValueError(
                f"Errore apertura video: {percorso_video}"
            )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            cap.release()
            raise ValueError(
                f"Il video non contiene frame validi: {percorso_video}"
            )

        indici = self.crea_indici_uniformi(total_frames)

        # Set per rendere veloce il controllo:
        # current_index in indici_da_estrarre
        indici_da_estrarre = set(indici)

        frames = []
        mask = []

        current_index = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if current_index in indici_da_estrarre:
                frame = cv2.resize(
                    frame,
                    (self.img_size, self.img_size),
                    interpolation=cv2.INTER_LINEAR
                )

                # OpenCV legge BGR; salviamo in RGB
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                frames.append(frame)
                mask.append(1)

            current_index += 1

            # Abbiamo già estratto tutti i frame richiesti
            if len(frames) == len(indici):
                break

        cap.release()

        if len(frames) == 0:
            raise ValueError(
                f"Nessun frame estratto dal video: {percorso_video}"
            )

        # Padding finale per video con meno di max_frame frame
        while len(frames) < self.max_frame:
            zero_frame = np.zeros(
                (self.img_size, self.img_size, 3),
                dtype=np.uint8
            )

            frames.append(zero_frame)
            mask.append(0)

        frames = np.asarray(
            frames,
            dtype=np.uint8
        )

        mask = np.asarray(
            mask,
            dtype=np.uint8
        )

        if frames.shape[0] != self.max_frame:
            raise RuntimeError(
                f"Numero frame errato per {percorso_video}: "
                f"attesi={self.max_frame}, ottenuti={frames.shape[0]}"
            )

        if mask.shape[0] != self.max_frame:
            raise RuntimeError(
                f"Numero valori mask errato per {percorso_video}: "
                f"attesi={self.max_frame}, ottenuti={mask.shape[0]}"
            )

        return frames, mask, total_frames

    def __call__(self, percorso_video):
        return self.estrai_frame_da_video(percorso_video)