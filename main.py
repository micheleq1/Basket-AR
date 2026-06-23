import os
import csv
import time
import math
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from rfdetr import RFDETRLarge
from movinets import MoViNet
from movinets.config import _C



VIDEO_PATH = r"/home/vrlab/Scrivania/BasketAR/dataset/dataset_hd/Copia di Copia di PSA_converted.mp4"
RFDETR_CHECKPOINT = r"/home/vrlab/Scrivania/BasketAR/Gruppo19/dataset/checkpoint_best_regular.pth"
ACTION_MODEL_CHECKPOINT = r"/home/vrlab/Scrivania/BasketAR/Gruppo19/dataset/best_multitask_basket_movinet_live_896.pth"

OUTPUT_CSV_RAW = r"/home/vrlab/Scrivania/BasketAR/Gruppo19/dataset/report_partita_raw.csv"
OUTPUT_CSV_EVENTI = r"/home/vrlab/Scrivania/BasketAR/Gruppo19/dataset/report_partita_eventi.csv"

INPUT_SIZE = 896
MAX_FRAMES = 32

# Limite massimo di durata del video da elaborare, in secondi.
# Impostare a None per elaborare il video per intero.
MAX_DURATION_SECONDS = 600

# ============================================================
# FINESTRATURA MULTI-SCALA
# ============================================================
#
# Problema risolto: con un'unica finestra da 2.0s i passaggi (durata
# media 0.5s) vengono "diluiti" tra molto contesto circostante e il
# modello tende a classificarli come idle/non-gioco. Una finestra
# corta invece taglia a metà i tiri lunghi (fino a 3.6s) e perde
# l'esito (segnato/sbagliato).
#
# Soluzione: due finestre in parallelo che terminano sempre nello
# stesso istante (la fine corrisponde al frame corrente).
#   - SHORT: precisa per passaggio/idle, poca diluizione
#   - LONG:  ampia abbastanza da contenere l'intero tiro + esito
#
# Ad ogni step viene scelta la predizione della finestra più adatta
# in base a una semplice regola (vedi fondi_predizioni_multiscala).

SHORT_CLIP_SECONDS = 0.8
LONG_CLIP_SECONDS = 2.0

# Ogni quanti secondi viene prodotta una predizione (fusa).
STEP_SECONDS = 0.4

# Soglia di confidenza sotto la quale una predizione viene considerata
# incerta e trattata come "no_action" nel report finale.
CONFIDENCE_THRESHOLD_ACTION = 0.75
CONFIDENCE_THRESHOLD_OUTCOME = 0.50

RFDETR_CONF_THRESHOLD = 0.40

CLASS_ID_PALLA = 0
CLASS_ID_CANESTRO = 1

CLASSI_TIRO = {"tiroDaDue", "tiroDaTre", "tiroLibero"}
CLASSI_DA_NON_RIPORTARE = {"idle", "non-gioco", "non_gioco", "no_action"}

USE_AMP = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MoViNetRFDetrLiveFusion(nn.Module):
    def __init__(
        self,
        movinet_out_features=640,
        rfdetr_size=19,
        rfdetr_encoded_size=128,
        gru_hidden_size=256,
        num_action_classes=5
    ):
        super().__init__()

        self.video_backbone = MoViNet(
            _C.MODEL.MoViNetA2,
            causal=False,
            pretrained=True
        )

        self.video_backbone.classifier = nn.Identity()

        for param in self.video_backbone.parameters():
            param.requires_grad = False

        for param in self.video_backbone.blocks[-1].parameters():
            param.requires_grad = True

        self.rfdetr_encoder = nn.Sequential(
            nn.LayerNorm(rfdetr_size),
            nn.Linear(rfdetr_size, rfdetr_encoded_size),
            nn.ReLU(),
            nn.Dropout(0.10)
        )

        self.rfdetr_gru = nn.GRU(
            input_size=rfdetr_encoded_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        total_fusion_dim = movinet_out_features + (gru_hidden_size * 2)

        self.fusion_norm = nn.LayerNorm(total_fusion_dim)

        self.head_action = nn.Sequential(
            nn.Linear(total_fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, num_action_classes)
        )

        self.head_outcome = nn.Sequential(
            nn.Linear(total_fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 2)
        )

    def forward(self, video_frames, rfdetr_features, mask):
        x = video_frames.permute(0, 4, 1, 2, 3).float() / 255.0
        B, C, T, H, W = x.shape

        x = x.reshape(B * T, C, H, W)
        x = torch.nn.functional.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )
        x = x.reshape(B, C, T, 224, 224)

        movinet_feat = self.video_backbone(x)

        if movinet_feat.dim() == 5:
            movinet_feat = movinet_feat.mean(dim=[2, 3, 4])

        B_rf, T_rf, _ = rfdetr_features.shape

        rf_enc = self.rfdetr_encoder(rfdetr_features)

        lengths = mask.to(dtype=torch.bool).sum(dim=1)
        lengths = lengths.clamp(min=1, max=T_rf).cpu()

        packed_rf = pack_padded_sequence(
            rf_enc,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        _, hidden = self.rfdetr_gru(packed_rf)

        rf_geom_vector = torch.cat(
            [hidden[-2], hidden[-1]],
            dim=1
        )

        fused_context = torch.cat(
            [movinet_feat, rf_geom_vector],
            dim=-1
        )

        fused_context = self.fusion_norm(fused_context)

        action_logits = self.head_action(fused_context)
        outcome_logits = self.head_outcome(fused_context)

        return action_logits, outcome_logits


def sec_to_timestamp(seconds):
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60

    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"

    return f"{m:02d}:{s:05.2f}"


def resize_square_bgr(frame_bgr, img_size):
    return cv2.resize(
        frame_bgr,
        (img_size, img_size),
        interpolation=cv2.INTER_LINEAR
    )


def crea_indici_uniformi(num_frames, max_frames):
    if num_frames <= 0:
        return []

    if num_frames <= max_frames:
        return list(range(num_frames))

    return np.linspace(
        0,
        num_frames - 1,
        max_frames,
        dtype=np.int64
    ).tolist()


def prendi_detection_migliore(detections, class_id_target):
    if detections is None or len(detections) == 0:
        return None, 0.0

    best_box = None
    best_conf = 0.0

    for box, class_id, conf in zip(
        detections.xyxy,
        detections.class_id,
        detections.confidence
    ):
        if int(class_id) == class_id_target:
            conf = float(conf)

            if conf > best_conf:
                best_conf = conf
                best_box = box

    return best_box, best_conf


def box_to_feature(box, conf, frame_w, frame_h):
    if box is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    x1, y1, x2, y2 = box

    centro_x = ((x1 + x2) / 2.0) / frame_w
    centro_y = ((y1 + y2) / 2.0) / frame_h

    larghezza = (x2 - x1) / frame_w
    altezza = (y2 - y1) / frame_h

    return [
        1.0,
        float(centro_x),
        float(centro_y),
        float(larghezza),
        float(altezza),
        float(conf)
    ]


def estrai_feature_frame(frame_rgb, detections):
    frame_h, frame_w = frame_rgb.shape[:2]

    palla_box, palla_conf = prendi_detection_migliore(
        detections,
        CLASS_ID_PALLA
    )

    canestro_box, canestro_conf = prendi_detection_migliore(
        detections,
        CLASS_ID_CANESTRO
    )

    palla_feature = box_to_feature(
        palla_box,
        palla_conf,
        frame_w,
        frame_h
    )

    canestro_feature = box_to_feature(
        canestro_box,
        canestro_conf,
        frame_w,
        frame_h
    )

    palla_presente = palla_feature[0]
    canestro_presente = canestro_feature[0]

    palla_x = palla_feature[1]
    palla_y = palla_feature[2]

    canestro_x = canestro_feature[1]
    canestro_y = canestro_feature[2]

    if palla_presente == 1.0 and canestro_presente == 1.0:
        dx = palla_x - canestro_x
        dy = palla_y - canestro_y
        distanza = np.sqrt(dx ** 2 + dy ** 2)
    else:
        dx = 0.0
        dy = 0.0
        distanza = 0.0

    feature_finale = (
        palla_feature
        + canestro_feature
        + [
            float(dx),
            float(dy),
            float(distanza)
        ]
    )

    return np.asarray(
        feature_finale,
        dtype=np.float32
    )


def estrai_feature_base15_runtime(frame_rgb, rfdetr_model):
    detections = rfdetr_model.predict(
        frame_rgb,
        threshold=RFDETR_CONF_THRESHOLD,
        shape=(INPUT_SIZE, INPUT_SIZE),
        include_source_image=False
    )

    return estrai_feature_frame(
        frame_rgb,
        detections
    )


def aggiungi_velocita_palla(sequence15):
    num_frames = sequence15.shape[0]

    velocita = np.zeros(
        (num_frames, 4),
        dtype=np.float32
    )

    for i in range(1, num_frames):
        palla_presente_ora = sequence15[i, 0]
        palla_presente_prima = sequence15[i - 1, 0]

        if palla_presente_ora == 1.0 and palla_presente_prima == 1.0:
            x_ora = sequence15[i, 1]
            y_ora = sequence15[i, 2]

            x_prima = sequence15[i - 1, 1]
            y_prima = sequence15[i - 1, 2]

            vx = x_ora - x_prima
            vy = y_ora - y_prima
            speed = np.sqrt(vx ** 2 + vy ** 2)

            velocita[i, 0] = vx
            velocita[i, 1] = vy
            velocita[i, 2] = speed
            velocita[i, 3] = 1.0

    return np.concatenate(
        [sequence15, velocita],
        axis=1
    ).astype(np.float32)


def inizializza_report_raw(output_csv):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "clip_index",
            "start_sec",
            "end_sec",
            "start_timestamp",
            "end_timestamp",
            "scala_usata",
            "azione_predetta",
            "confidenza_azione",
            "outcome_predetto",
            "confidenza_outcome",
            "azione_short",
            "conf_short",
            "azione_long",
            "conf_long",
            "tempo_elaborazione_sec"
        ])


def aggiorna_report_raw(output_csv, row):
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            row["clip_index"],
            f"{row['start_sec']:.3f}",
            f"{row['end_sec']:.3f}",
            row["start_timestamp"],
            row["end_timestamp"],
            row["scala_usata"],
            row["azione_predetta"],
            f"{row['confidenza_azione']:.6f}",
            row["outcome_predetto"],
            f"{row['confidenza_outcome']:.6f}",
            row["azione_short"],
            f"{row['conf_short']:.6f}",
            row["azione_long"],
            f"{row['conf_long']:.6f}",
            f"{row['tempo_elaborazione_sec']:.3f}"
        ])

        f.flush()


# ============================================================
# FUSIONE MULTI-SCALA CON ISTERESI
# ============================================================
#
# Problema risolto: con una soglia fissa unica, se la confidenza di
# LONG oscilla intorno alla soglia step dopo step (es. 0.56, 0.52,
# 0.58, 0.49...) il sistema flippa avanti e indietro tra "tiro" (da
# LONG) e "passaggio" (da SHORT) SULLA STESSA azione reale, producendo
# due eventi sovrapposti per un'unica azione invece di uno solo.
#
# Soluzione: due soglie (isteresi). Per passare da SHORT a LONG serve
# superare la soglia ALTA; una volta su LONG, ci si resta fino a che
# la confidenza non scende sotto la soglia BASSA, molto più permissiva.
# Questo rende la decisione "sticky" e assorbe le piccole oscillazioni
# attorno al punto di soglia.
SOGLIA_ATTIVAZIONE_LONG = 0.80
SOGLIA_DISATTIVAZIONE_LONG = 0.65


class FusoreMultiScala:
    def __init__(
        self,
        soglia_attivazione=SOGLIA_ATTIVAZIONE_LONG,
        soglia_disattivazione=SOGLIA_DISATTIVAZIONE_LONG
    ):
        self.scala_attiva = "short"
        self.soglia_attivazione = soglia_attivazione
        self.soglia_disattivazione = soglia_disattivazione

    def fondi(self, pred_short, pred_long):
        azione_long = pred_long["azione"]
        conf_long = pred_long["conf_azione"]

        long_propone_tiro = azione_long in CLASSI_TIRO

        if self.scala_attiva == "short":
            if long_propone_tiro and conf_long >= self.soglia_attivazione:
                self.scala_attiva = "long"
        else:
            # scala_attiva == "long": resta su long finché la
            # confidenza non scende sotto la soglia bassa, oppure
            # finché long non propone più un tiro.
            if not long_propone_tiro or conf_long < self.soglia_disattivazione:
                self.scala_attiva = "short"

        if self.scala_attiva == "long" and long_propone_tiro:
            return {
                "scala_usata": "long",
                "azione": azione_long,
                "conf_azione": conf_long,
                "outcome": pred_long["outcome"],
                "conf_outcome": pred_long["conf_outcome"]
            }

        return {
            "scala_usata": "short",
            "azione": pred_short["azione"],
            "conf_azione": pred_short["conf_azione"],
            "outcome": pred_short["outcome"],
            "conf_outcome": pred_short["conf_outcome"]
        }


# ============================================================
# SCELTA OUTCOME TRA FINESTRE IN CONFLITTO
# ============================================================
#
# All'interno di uno stesso evento (es. un tiro coperto da più
# finestre consecutive) le finestre possono disaccordare
# sull'outcome (segnato/non-segnato). Prendere semplicemente la
# confidenza massima è fragile: una differenza tipo 1.000 vs 0.990
# non è significativa, ma "vince" comunque la prima per via
# dell'ordine di valutazione.
#
# Questa funzione preferisce la finestra temporalmente più recente
# quando le confidenze sono vicine (entro MARGINE_PAREGGIO_OUTCOME),
# perché una finestra che termina più avanti nel tempo ha visto più
# contesto del momento in cui la palla entra o esce dal canestro.
MARGINE_PAREGGIO_OUTCOME = 1.000


def scegli_outcome_migliore(finestre_tiro):
    finestre_ordinate = sorted(
        finestre_tiro,
        key=lambda f: f["confidenza_outcome"],
        reverse=True
    )

    migliore = finestre_ordinate[0]

    for f in finestre_ordinate[1:]:
        differenza = migliore["confidenza_outcome"] - f["confidenza_outcome"]

        if differenza <= MARGINE_PAREGGIO_OUTCOME:
            # Confidenze quasi pari: preferisci la finestra più recente.
            if f["end_sec"] > migliore["end_sec"]:
                migliore = f
        else:
            # Differenza significativa: la finestra ordinata per prima
            # resta la scelta migliore, non serve guardare oltre.
            break

    return migliore["outcome_predetto"], migliore["confidenza_outcome"]


# ============================================================
# MERGE EVENTI-TIRO SOVRAPPOSTI (RISOLVE L'OSCILLAZIONE TRA
# SOTTOTIPI DI TIRO SULLO STESSO TIRO REALE)
# ============================================================
#
# Problema osservato: la finestra LONG, scorrendo di pochi sample
# alla volta, vede un contenuto quasi identico da uno step al
# successivo. A volte questo piccolo cambiamento è sufficiente per
# far "flippare" la sottoclasse del tiro (es. tiroDaDue -> tiroLibero
# -> tiroDaDue), perché le tre classi di tiro sono visivamente simili.
# Il raggruppamento per azione consecutiva tratta ogni flip come un
# evento NUOVO, producendo più eventi che si sovrappongono nel tempo
# pur essendo, in realtà, lo stesso tiro.
#
# Soluzione: dopo il raggruppamento per azione consecutiva, gli
# eventi di tipo tiro (tiroDaDue/tiroDaTre/tiroLibero) che si
# sovrappongono temporalmente vengono uniti in un unico evento. La
# sottoclasse finale viene decisa con un voto pesato sulla somma delle
# confidenze tra tutte le finestre coinvolte (più "evidenza" vince),
# invece di scegliere semplicemente il primo flip rilevato.
#
# Gli eventi non-tiro (passaggio/idle/non-gioco) NON vengono uniti
# tra loro da questa funzione, anche se si sovrappongono leggermente
# con un evento-tiro adiacente: quella sovrapposizione è quasi sempre
# dovuta al "guardare indietro" della finestra LONG dentro l'azione
# precedente, non a un errore di tipo da correggere.
def unisci_gruppi_tiro_sovrapposti(gruppi):
    gruppi_tiro = [g for g in gruppi if g["azione"] in CLASSI_TIRO]
    gruppi_non_tiro = [g for g in gruppi if g["azione"] not in CLASSI_TIRO]

    gruppi_tiro.sort(key=lambda g: g["start_sec"])

    supergruppi = []
    corrente = None

    for g in gruppi_tiro:
        if corrente is None:
            corrente = {
                "start_sec": g["start_sec"],
                "end_sec": g["end_sec"],
                "finestre": list(g["finestre"])
            }
            continue

        # Sovrapposizione reale tra l'intervallo del gruppo e quello
        # già accumulato nel supergruppo corrente.
        if g["start_sec"] <= corrente["end_sec"]:
            corrente["end_sec"] = max(corrente["end_sec"], g["end_sec"])
            corrente["finestre"].extend(g["finestre"])
        else:
            supergruppi.append(corrente)
            corrente = {
                "start_sec": g["start_sec"],
                "end_sec": g["end_sec"],
                "finestre": list(g["finestre"])
            }

    if corrente is not None:
        supergruppi.append(corrente)

    # I supergruppi non hanno ancora un'azione definitiva: verrà
    # decisa più avanti con il voto pesato in calcola_evento_finale.
    for sg in supergruppi:
        sg["azione"] = None

    return gruppi_non_tiro + supergruppi


# ============================================================
# CALCOLO AZIONE E OUTCOME FINALI DI UN GRUPPO/SUPERGRUPPO
# ============================================================
#
# Generalizzato per funzionare sia sui gruppi "normali" (azione
# unica, prodotti dal raggruppamento per azione consecutiva) sia sui
# supergruppi-tiro (più sottotipi di tiro mescolati dopo il merge
# delle sovrapposizioni): il voto pesato per l'azione si comporta in
# modo identico nei due casi, perché un gruppo omogeneo ha banalmente
# un solo candidato e quindi vince sempre quello.
def calcola_evento_finale(gruppo):
    finestre = gruppo["finestre"]

    # Voto pesato: per ogni azione candidata tra le finestre, somma
    # le rispettive confidenze. Vince l'azione con più "evidenza"
    # totale, non semplicemente la prima rilevata.
    #
    # IMPORTANTE: la soglia di confidenza va applicata qui, PER OGNI
    # SINGOLA FINESTRA, prima di accumulare il voto. Applicarla solo
    # a posteriori sull'azione vincente (e solo per le classi non-tiro,
    # come faceva la versione precedente) permette a una finestra
    # isolata e poco sicura di "risuscitare" come evento tiro valido:
    # se è l'unico voto nel suo gruppo vince per assenza di concorrenza,
    # e il controllo a posteriori la esentava comunque dalla soglia
    # solo perché la sua etichetta è una classe di tiro.
    somma_conf_per_azione = {}
    conf_max_per_azione = {}

    for f in finestre:
        raw_az = f["azione_predetta"]
        conf = f["confidenza_azione"]

        if conf < CONFIDENCE_THRESHOLD_ACTION:
            az = "no_action"
        else:
            az = raw_az

        somma_conf_per_azione[az] = somma_conf_per_azione.get(az, 0.0) + conf
        conf_max_per_azione[az] = max(conf_max_per_azione.get(az, 0.0), conf)

    azione_finale = max(
        somma_conf_per_azione,
        key=lambda az: somma_conf_per_azione[az]
    )

    conf_azione_finale = conf_max_per_azione[azione_finale]

    is_tiro = azione_finale in CLASSI_TIRO

    if is_tiro:
        finestre_tiro = [
            f for f in finestre
            if f["outcome_predetto"] != "non_applicabile"
        ]

        if len(finestre_tiro) > 0:
            outcome, conf_outcome = scegli_outcome_migliore(
                finestre_tiro
            )

            if conf_outcome < CONFIDENCE_THRESHOLD_OUTCOME:
                outcome = "incerto"
        else:
            outcome = "incerto"
            conf_outcome = 0.0
    else:
        outcome = "non_applicabile"
        conf_outcome = 0.0

    return {
        "azione": azione_finale,
        "start_sec": gruppo["start_sec"],
        "end_sec": gruppo["end_sec"],
        "durata_sec": gruppo["end_sec"] - gruppo["start_sec"],
        "confidenza_azione": conf_azione_finale,
        "outcome": outcome,
        "confidenza_outcome": conf_outcome,
        "num_finestre": len(finestre)
    }


# ============================================================
# POST-PROCESSING: UNIONE FINESTRE SOVRAPPOSTE IN EVENTI
# ============================================================

# ============================================================
# SOPPRESSIONE PASSAGGI SHORT SOVRAPPOSTI A TIRI LONG
# ============================================================
#
# Questa funzione NON modifica la logica di inferenza o di fusione.
# Viene applicata soltanto prima della generazione del file eventi.
#
# Regola:
#   - se una finestra SHORT ha rilevato "passaggio";
#   - se l'intervallo temporale della SHORT si sovrappone a quello
#     di una finestra LONG che ha rilevato un tiro con confidenza
#     almeno pari a CONFIDENCE_THRESHOLD_ACTION;
#   - e la predizione fusa salvata è ancora "passaggio";
#
# allora la predizione viene trasformata in "no_action", così non
# viene inserita nel report finale degli eventi.
def sopprimi_passaggi_short_sovrapposti_a_tiri_long(predizioni_raw):
    intervalli_tiro_long = []

    for pred in predizioni_raw:
        azione_long = pred.get("azione_long")
        conf_long = float(pred.get("conf_long", 0.0))

        if (
            azione_long in CLASSI_TIRO
            and conf_long >= CONFIDENCE_THRESHOLD_ACTION
        ):
            intervalli_tiro_long.append(
                (
                    float(pred["start_sec_long"]),
                    float(pred["end_sec_long"])
                )
            )

    passaggi_soppressi = 0

    for pred in predizioni_raw:
        if pred.get("azione_predetta") != "passaggio":
            continue

        if pred.get("azione_short") != "passaggio":
            continue

        short_start = float(pred["start_sec_short"])
        short_end = float(pred["end_sec_short"])

        sovrapposto_a_tiro_long = any(
            max(short_start, long_start) < min(short_end, long_end)
            for long_start, long_end in intervalli_tiro_long
        )

        if sovrapposto_a_tiro_long:
            pred["azione_predetta"] = "no_action"
            pred["outcome_predetto"] = "non_applicabile"
            pred["confidenza_outcome"] = 0.0
            passaggi_soppressi += 1

    return passaggi_soppressi


def unisci_predizioni_in_eventi(predizioni_raw):
    gruppi = []
    gruppo_corrente = None

    for pred in predizioni_raw:
        azione = pred["azione_predetta"]
        conf_azione = pred["confidenza_azione"]

        if conf_azione < CONFIDENCE_THRESHOLD_ACTION:
            azione = "no_action"

        if gruppo_corrente is None:
            gruppo_corrente = {
                "azione": azione,
                "start_sec": pred["start_sec"],
                "end_sec": pred["end_sec"],
                "finestre": [pred]
            }
            continue

        if azione == gruppo_corrente["azione"]:
            gruppo_corrente["end_sec"] = pred["end_sec"]
            gruppo_corrente["finestre"].append(pred)
        else:
            gruppi.append(gruppo_corrente)
            gruppo_corrente = {
                "azione": azione,
                "start_sec": pred["start_sec"],
                "end_sec": pred["end_sec"],
                "finestre": [pred]
            }

    if gruppo_corrente is not None:
        gruppi.append(gruppo_corrente)

    # Stadio 2: unisce gli eventi-tiro che si sovrappongono nel tempo
    # (oscillazione tra sottotipi sullo stesso tiro reale).
    gruppi = unisci_gruppi_tiro_sovrapposti(gruppi)

    # Stadio 3: calcola azione/outcome finali per ogni gruppo,
    # ordinando per start_sec dato che il merge può aver cambiato
    # l'ordine relativo tra gruppi-tiro e gruppi non-tiro.
    gruppi.sort(key=lambda g: g["start_sec"])

    eventi_finali = [
        calcola_evento_finale(g)
        for g in gruppi
    ]

    return eventi_finali


def salva_report_eventi(output_csv, eventi):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "evento_index",
            "azione",
            "start_sec",
            "end_sec",
            "start_timestamp",
            "end_timestamp",
            "durata_sec",
            "confidenza_azione",
            "outcome",
            "confidenza_outcome",
            "num_finestre_unite"
        ])

        evento_index = 0

        for evento in eventi:
            if evento["azione"] in CLASSI_DA_NON_RIPORTARE:
                continue

            writer.writerow([
                evento_index,
                evento["azione"],
                f"{evento['start_sec']:.3f}",
                f"{evento['end_sec']:.3f}",
                sec_to_timestamp(evento["start_sec"]),
                sec_to_timestamp(evento["end_sec"]),
                f"{evento['durata_sec']:.3f}",
                f"{evento['confidenza_azione']:.6f}",
                evento["outcome"],
                f"{evento['confidenza_outcome']:.6f}",
                evento["num_finestre"]
            ])

            evento_index += 1


def carica_rfdetr():
    checkpoint = Path(RFDETR_CHECKPOINT)

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint RF-DETR non trovato: {checkpoint}"
        )

    print("Carico RF-DETR Large:", checkpoint)

    model = RFDETRLarge(
        pretrain_weights=str(checkpoint),
        num_classes=2
    )

    if hasattr(model, "remove_optimized_model"):
        try:
            model.remove_optimized_model()
        except Exception:
            pass

    return model


def carica_modello_finale(device):
    checkpoint = Path(ACTION_MODEL_CHECKPOINT)

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint modello finale non trovato: {checkpoint}"
        )

    print("Carico modello finale:", checkpoint)

    ckpt = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False
    )

    action_to_idx = ckpt["action_to_idx"]
    idx_to_action = ckpt["idx_to_action"]

    outcome_to_idx = ckpt["outcome_to_idx"]
    idx_to_outcome = ckpt["idx_to_outcome"]

    model = MoViNetRFDetrLiveFusion(
        movinet_out_features=ckpt.get("movinet_size", 640),
        rfdetr_size=ckpt.get("rfdetr_size", 19),
        rfdetr_encoded_size=ckpt.get("rfdetr_encoded_size", 128),
        gru_hidden_size=ckpt.get("hidden_size", 256),
        num_action_classes=ckpt.get("num_action_classes", len(action_to_idx))
    ).to(device)

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.eval()

    return model, idx_to_action, idx_to_outcome


def costruisci_clip_da_lista(items, max_frames=32):
    """
    items: lista di tuple (frame_index, frame_rgb, feat15), già
    ritagliata alla finestra desiderata (SHORT o LONG).

    Esegue il sampling uniforme a max_frames, identico a quanto
    fatto in training, indipendentemente dalla durata della finestra
    di provenienza.
    """

    indici = crea_indici_uniformi(
        len(items),
        max_frames
    )

    frames = []
    mask = []
    feature15 = []

    for i in indici:
        _, frame_rgb, feat15 = items[i]

        frames.append(frame_rgb)
        feature15.append(feat15)
        mask.append(1)

    if len(frames) == 0:
        frames.append(
            np.zeros(
                (INPUT_SIZE, INPUT_SIZE, 3),
                dtype=np.uint8
            )
        )
        feature15.append(
            np.zeros(
                15,
                dtype=np.float32
            )
        )
        mask.append(0)

    # IMPORTANTE: il padding deve essere identico a quello usato in
    # training (video_preprocessor.py), che usa frame NERI (zeri) con
    # mask=0, non l'ultimo frame duplicato. Per le finestre corte
    # (es. SHORT da 0.7s, che spesso ha solo ~21 frame reali su 32
    # richiesti) la differenza tra frame neri e frame duplicati è
    # enorme: con la duplicazione il modello vede "movimento congelato"
    # invece del segnale di padding che ha imparato a ignorare.
    zero_frame = np.zeros(
        (INPUT_SIZE, INPUT_SIZE, 3),
        dtype=np.uint8
    )

    zero_feat = np.zeros(
        15,
        dtype=np.float32
    )

    while len(frames) < max_frames:
        frames.append(zero_frame.copy())
        feature15.append(zero_feat.copy())
        mask.append(0)

    frames = np.asarray(
        frames[:max_frames],
        dtype=np.uint8
    )

    mask = np.asarray(
        mask[:max_frames],
        dtype=np.uint8
    )

    feature15 = np.asarray(
        feature15[:max_frames],
        dtype=np.float32
    )

    rfdetr_features = aggiungi_velocita_palla(
        feature15
    )

    return frames, mask, rfdetr_features


def predici_clip(frames_rgb, mask, rfdetr_features, model, device):
    frames_tensor = torch.from_numpy(
        frames_rgb
    ).unsqueeze(0).to(
        device,
        non_blocking=True
    )

    mask_tensor = torch.from_numpy(
        mask
    ).unsqueeze(0).long().to(
        device,
        non_blocking=True
    )

    rfdetr_tensor = torch.from_numpy(
        rfdetr_features
    ).unsqueeze(0).float().to(
        device,
        non_blocking=True
    )

    amp_enabled = USE_AMP and device.type == "cuda"

    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled
        ):
            action_logits, outcome_logits = model(
                frames_tensor,
                rfdetr_tensor,
                mask_tensor
            )

        action_prob = torch.softmax(
            action_logits,
            dim=1
        )[0]

        outcome_prob = torch.softmax(
            outcome_logits,
            dim=1
        )[0]

    action_idx = int(
        torch.argmax(action_prob).item()
    )

    outcome_idx = int(
        torch.argmax(outcome_prob).item()
    )

    action_conf = float(
        action_prob[action_idx].item()
    )

    outcome_conf = float(
        outcome_prob[outcome_idx].item()
    )

    action_prob_np = action_prob.detach().cpu().numpy()

    return action_idx, action_conf, outcome_idx, outcome_conf, action_prob_np


# ============================================================
# FILTRO EURISTICO PASSAGGIO/PALLEGGIO
# ============================================================
#
# Causa di fondo (nei dati, non risolvibile dal solo codice di
# inferenza): le clip di training etichettate "passaggio" mostrano
# sempre il gesto del passaggio, mai un palleggio isolato come
# esempio negativo. Il modello non ha mai imparato a distinguere
# esplicitamente i due casi, e su una finestra breve come SHORT
# (0.7s) un singolo palleggio è visivamente simile a un passaggio.
#
# La correzione corretta sul lungo termine è arricchire il dataset
# di training con clip di palleggio etichettate "idle". Questo
# filtro è solo un correttivo euristico nel frattempo: un vero
# passaggio sposta la palla rapidamente verso un punto diverso,
# mentre un palleggio la fa oscillare nello stesso punto. Se la
# velocità media della palla nella finestra è troppo bassa, il
# "passaggio" rilevato viene declassato a "idle".
#
# Indici nel vettore rfdetr_features (19 valori):
#   17 = velocita_palla (modulo, normalizzato 0-1 per step)
#   18 = velocita_valida (1.0 se calcolata su due frame consecutivi
#        entrambi con palla rilevata, 0.0 altrimenti)
#
# ATTENZIONE: questa soglia è un valore di partenza, va calibrata
# osservando la colonna corrispondente nei log/CSV su qualche
# partita campione (passaggi veri vs palleggi mal classificati).
VELOCITA_MINIMA_PASSAGGIO = 0.010


def velocita_media_palla(rfdetr_features):
    valida = rfdetr_features[:, 18] > 0.0

    if not np.any(valida):
        return 0.0

    return float(rfdetr_features[valida, 17].mean())


def esegui_predizione_scala(
    items,
    model,
    device,
    idx_to_action,
    idx_to_outcome,
    classi_escluse=None,
    filtra_palleggio=False
):
    """
    Esegue il sampling + l'inferenza su una lista di sample già
    ritagliata alla finestra desiderata (SHORT o LONG) e ritorna
    un dizionario con azione/outcome predetti per quella scala.

    classi_escluse: insieme di nomi di azione che questa scala non
    può MAI restituire (es. i tiri per la finestra SHORT, troppo
    breve per giudicarli, oppure "passaggio" per la finestra LONG,
    troppo diluita per quell'azione breve). Se l'argmax originale
    del modello cade su una classe esclusa, viene ricalcolato
    l'argmax tra le sole classi ammesse, così ogni scala può
    riportare solo azioni del tipo che è strutturalmente in grado
    di giudicare.

    filtra_palleggio: se True e l'azione risultante è "passaggio",
    verifica la velocità media della palla nella finestra. Se troppo
    bassa (palla che oscilla sul posto, tipico del palleggio),
    declassa l'azione a "idle".
    """

    if classi_escluse is None:
        classi_escluse = set()

    frames_rgb, mask, rfdetr_features = costruisci_clip_da_lista(
        items,
        max_frames=MAX_FRAMES
    )

    action_idx, action_conf, outcome_idx, outcome_conf, action_prob_np = predici_clip(
        frames_rgb,
        mask,
        rfdetr_features,
        model,
        device
    )

    azione = idx_to_action[action_idx]

    if azione in classi_escluse:
        # L'argmax originale è su una classe non ammessa per questa
        # scala: ricalcola l'argmax considerando solo le classi
        # ammesse, mascherando le altre.
        indici_ammessi = [
            i for i in range(len(action_prob_np))
            if idx_to_action[i] not in classi_escluse
        ]

        idx_migliore = max(
            indici_ammessi,
            key=lambda i: action_prob_np[i]
        )

        azione = idx_to_action[idx_migliore]
        action_conf = float(action_prob_np[idx_migliore])

    if filtra_palleggio and azione == "passaggio":
        velocita = velocita_media_palla(rfdetr_features)

        if velocita < VELOCITA_MINIMA_PASSAGGIO:
            azione = "idle"

            action_to_idx_inv = {
                v: k for k, v in idx_to_action.items()
            }
            idx_idle = action_to_idx_inv.get("idle")

            if idx_idle is not None:
                action_conf = float(action_prob_np[idx_idle])
            else:
                action_conf = 0.50

    if azione in CLASSI_TIRO:
        outcome = idx_to_outcome[outcome_idx]
    else:
        outcome = "non_applicabile"

    return {
        "azione": azione,
        "conf_azione": action_conf,
        "outcome": outcome,
        "conf_outcome": outcome_conf
    }


def main():
    video_path = Path(VIDEO_PATH)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video non trovato: {video_path}"
        )

    device = torch.device(DEVICE)

    print("Device:", device)
    print("Video:", video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Impossibile aprire il video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    durata = total_frames / fps

    print(f"FPS: {fps:.3f}")
    print(f"Frame totali: {total_frames}")
    print(f"Durata: {sec_to_timestamp(durata)}")

    if MAX_DURATION_SECONDS is not None:
        max_frames_da_processare = int(MAX_DURATION_SECONDS * fps)
        max_frames_da_processare = min(max_frames_da_processare, total_frames)

        print(
            f"Limite impostato: primi {MAX_DURATION_SECONDS:.1f}s "
            f"({sec_to_timestamp(MAX_DURATION_SECONDS)}) "
            f"-> {max_frames_da_processare} frame su {total_frames}"
        )
    else:
        max_frames_da_processare = total_frames
        print("Nessun limite di durata: elaboro il video per intero.")

    # ========================================================
    # CALCOLO PARAMETRI MULTI-SCALA
    # ========================================================
    #
    # Il buffer denso campiona ad una frequenza sufficiente a
    # popolare 32 frame distinti anche nella finestra SHORT (la
    # più corta). In pratica, con SHORT_CLIP_SECONDS=0.7s questo
    # richiede quasi sempre di campionare ogni singolo frame video.
    dense_sample_rate = MAX_FRAMES / SHORT_CLIP_SECONDS
    dense_sample_interval_frames = max(
        1,
        int(round(fps / dense_sample_rate))
    )

    short_window_samples = max(
        1,
        int(math.ceil(SHORT_CLIP_SECONDS * fps / dense_sample_interval_frames))
    )

    long_window_samples = max(
        1,
        int(math.ceil(LONG_CLIP_SECONDS * fps / dense_sample_interval_frames))
    )

    step_samples = max(
        1,
        int(round(STEP_SECONDS * fps / dense_sample_interval_frames))
    )

    # Margine di sicurezza per evitare di restare a corto di sample
    # per la finestra LONG quando il timing video non è perfettamente
    # regolare.
    dense_buffer_maxlen = long_window_samples + step_samples + 5

    print(f"Finestra SHORT: {SHORT_CLIP_SECONDS}s ({short_window_samples} sample)")
    print(f"Finestra LONG:  {LONG_CLIP_SECONDS}s ({long_window_samples} sample)")
    print(f"Step predizione: {STEP_SECONDS}s ({step_samples} sample)")
    print(f"Campionamento denso ogni {dense_sample_interval_frames} frame video")
    print(f"Buffer denso: {dense_buffer_maxlen} sample")
    print(f"Report raw: {OUTPUT_CSV_RAW}")
    print(f"Report eventi: {OUTPUT_CSV_EVENTI}")

    rfdetr_model = carica_rfdetr()
    final_model, idx_to_action, idx_to_outcome = carica_modello_finale(
        device
    )

    inizializza_report_raw(
        OUTPUT_CSV_RAW
    )

    predizioni_raw = []

    fusore = FusoreMultiScala()

    dense_buffer = deque(
        maxlen=dense_buffer_maxlen
    )

    frame_index = 0
    dense_sample_index = 0
    clip_index = 0
    next_prediction_sample_index = short_window_samples

    start_total = time.time()

    while True:
        if frame_index >= max_frames_da_processare:
            print(
                f"\nRaggiunto il limite di {MAX_DURATION_SECONDS:.1f}s "
                f"({frame_index} frame). Interrompo la lettura."
            )
            break

        ret, frame_bgr = cap.read()

        if not ret:
            break

        if frame_index % dense_sample_interval_frames == 0:
            frame_square_bgr = resize_square_bgr(
                frame_bgr,
                INPUT_SIZE
            )

            frame_rgb = cv2.cvtColor(
                frame_square_bgr,
                cv2.COLOR_BGR2RGB
            )

            feat15 = estrai_feature_base15_runtime(
                frame_rgb,
                rfdetr_model
            )

            dense_buffer.append(
                (
                    frame_index,
                    frame_rgb,
                    feat15
                )
            )

            dense_sample_index += 1

            buffer_pronto = len(dense_buffer) >= short_window_samples

            if (
                buffer_pronto
                and dense_sample_index >= next_prediction_sample_index
            ):
                loop_start = time.time()

                items = list(dense_buffer)

                clip_end_frame = items[-1][0]
                clip_end_sec = clip_end_frame / fps

                # Finestra SHORT: ultimi short_window_samples sample.
                items_short = items[-short_window_samples:]
                clip_start_sec_short = items_short[0][0] / fps

                pred_short = esegui_predizione_scala(
                    items_short,
                    final_model,
                    device,
                    idx_to_action,
                    idx_to_outcome,
                    classi_escluse=CLASSI_TIRO,
                    filtra_palleggio=True
                )

                # Finestra LONG: ultimi long_window_samples sample
                # (o tutti quelli disponibili se il buffer non è
                # ancora pieno, es. inizio video).
                n_long_disponibili = min(long_window_samples, len(items))
                items_long = items[-n_long_disponibili:]
                clip_start_sec_long = items_long[0][0] / fps

                pred_long = esegui_predizione_scala(
                    items_long,
                    final_model,
                    device,
                    idx_to_action,
                    idx_to_outcome,
                    classi_escluse={"passaggio"}
                )

                fused = fusore.fondi(
                    pred_short,
                    pred_long
                )

                # Lo start_sec riportato corrisponde alla finestra
                # effettivamente usata dalla fusione, per coerenza
                # con il post-processing che unisce eventi consecutivi.
                if fused["scala_usata"] == "long":
                    clip_start_sec = clip_start_sec_long
                else:
                    clip_start_sec = clip_start_sec_short

                elapsed = time.time() - loop_start

                row = {
                    "clip_index": clip_index,
                    "start_sec": clip_start_sec,
                    "end_sec": clip_end_sec,
                    "start_timestamp": sec_to_timestamp(clip_start_sec),
                    "end_timestamp": sec_to_timestamp(clip_end_sec),
                    "scala_usata": fused["scala_usata"],
                    "azione_predetta": fused["azione"],
                    "confidenza_azione": fused["conf_azione"],
                    "outcome_predetto": fused["outcome"],
                    "confidenza_outcome": fused["conf_outcome"],
                    "azione_short": pred_short["azione"],
                    "conf_short": pred_short["conf_azione"],
                    "azione_long": pred_long["azione"],
                    "conf_long": pred_long["conf_azione"],
                    # Intervalli reali delle due finestre, usati solo
                    # nel post-processing per rilevare le sovrapposizioni.
                    "start_sec_short": clip_start_sec_short,
                    "end_sec_short": clip_end_sec,
                    "start_sec_long": clip_start_sec_long,
                    "end_sec_long": clip_end_sec,
                    "tempo_elaborazione_sec": elapsed
                }

                aggiorna_report_raw(
                    OUTPUT_CSV_RAW,
                    row
                )

                predizioni_raw.append(row)

                print(
                    f"[clip {clip_index:05d}] "
                    f"{row['start_timestamp']} -> {row['end_timestamp']} | "
                    f"scala={fused['scala_usata']:<5s} | "
                    f"{fused['azione']} ({fused['conf_azione']:.3f}) | "
                    f"{fused['outcome']} ({fused['conf_outcome']:.3f}) | "
                    f"[short: {pred_short['azione']} {pred_short['conf_azione']:.2f}] "
                    f"[long: {pred_long['azione']} {pred_long['conf_azione']:.2f}] | "
                    f"{elapsed:.2f}s"
                )

                clip_index += 1
                next_prediction_sample_index += step_samples

        frame_index += 1

    cap.release()

    print("\nInferenza completata.")
    print(f"Clip elaborate: {clip_index}")
    print(f"Tempo totale: {time.time() - start_total:.2f}s")

    print("\nUnione finestre sovrapposte in eventi...")

    passaggi_soppressi = sopprimi_passaggi_short_sovrapposti_a_tiri_long(
        predizioni_raw
    )

    print(
        "Passaggi SHORT sovrapposti a tiri LONG convertiti "
        f"in no_action: {passaggi_soppressi}"
    )

    eventi = unisci_predizioni_in_eventi(predizioni_raw)

    salva_report_eventi(
        OUTPUT_CSV_EVENTI,
        eventi
    )

    eventi_rilevanti = [
        e for e in eventi
        if e["azione"] not in CLASSI_DA_NON_RIPORTARE
    ]

    print(f"Eventi totali (incluso idle/non-gioco): {len(eventi)}")
    print(f"Eventi rilevanti riportati nel CSV finale: {len(eventi_rilevanti)}")
    print(f"Report raw: {OUTPUT_CSV_RAW}")
    print(f"Report eventi: {OUTPUT_CSV_EVENTI}")


if __name__ == "__main__":
    main()