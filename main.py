import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import VideoDataset
import os
from torchvision import transforms

mobilenet_transforms = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])


def testDataset():
    
    FILE_ATTUALE = os.path.dirname(os.path.abspath(__file__))

    DATASET_CARTELLA = os.path.abspath(
        os.path.join(FILE_ATTUALE, "..", "dataset")
    )
    MANIFEST = os.path.abspath(
        os.path.join(DATASET_CARTELLA, "manifest.csv")
    )

    train_dataset = VideoDataset(
        annotations_file=MANIFEST,
        video_dir=DATASET_CARTELLA,
        split="train",
        maxFrame=32,   
        imgSize=224,
        transform=mobilenet_transforms  
    )

    validation_dataset = VideoDataset(
        annotations_file=MANIFEST,
        video_dir=DATASET_CARTELLA,
        split="val",
        maxFrame=32,   
        imgSize=224,
        transform=mobilenet_transforms  
    )   

    test_dataset = VideoDataset(
        annotations_file=MANIFEST,
        video_dir=DATASET_CARTELLA,
        split="test",
        maxFrame=32,   
        imgSize=224,
        transform=mobilenet_transforms  
    )

    # prendiamo le label del train 
    raw_labels = train_dataset.video_label.iloc[:, 5].values
    
    # convertiamo la colonna label di stringhe in numeri
    train_labels_numeric, class_names = pd.factorize(raw_labels)
    print(f"-> Classi rilevate: {list(class_names)}")

    # contiamo quanti video ci sono per ogni classe numerica
    class_count = np.bincount(train_labels_numeric)
    print(f"-> Distribuzione classi nel Train: {class_count}")

    # calcoliamo i pesi inversi
    class_weights = 1.0 / class_count
    
    # associamo il peso ad ogni video
    sample_weights = [class_weights[label] for label in train_labels_numeric]
    sample_weights = torch.DoubleTensor(sample_weights)
    
    # sampler pesato
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_dataloader = DataLoader(train_dataset, batch_size=8, sampler=sampler)
    val_dataloader   = DataLoader(validation_dataset,   batch_size=8, shuffle=False)
    test_dataloader  = DataLoader(test_dataset,  batch_size=8, shuffle=False)

    train_frames, train_masks, train_labels_out = next(iter(train_dataloader))
    b_train, t_train, c_train, h_train, w_train = train_frames.shape
    train_frames_per_mobilenet = train_frames.view(b_train * t_train, c_train, h_train, w_train)
    
    val_frames, val_masks, val_labels_out = next(iter(val_dataloader))
    b_val, t_val, c_val, h_val, w_val = val_frames.shape
    val_frames_per_mobilenet = val_frames.view(b_val * t_val, c_val, h_val, w_val)

    test_frames, test_masks, test_labels_out = next(iter(test_dataloader))
    b_test, t_test, c_test, h_test, w_test = test_frames.shape
    test_frames_per_mobilenet = test_frames.view(b_test * t_test, c_test, h_test, w_test)    

    #QUESTO NON L'HO CAPITO MA DAVA ERRORE E LO HA FATTO CHI NE SA 
    # Se train_labels_out è una tupla di stringhe/oggetti, la stampiamo semplicemente convertendola in lista
    if isinstance(train_labels_out, tuple):
        print(f"Label estratte in questo batch: {list(train_labels_out)}")
    else:
        # Se invece è un Tensor (numerico), usiamo il classico .tolist()
        print(f"Label estratte in questo batch: {train_labels_out.tolist()}")
        
    print("Caro Michele, funziona, perché le classi rare tipo tiroDaTre vengono prese spesso")

if __name__ == "__main__":
    testDataset()