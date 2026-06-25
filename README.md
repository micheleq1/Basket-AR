# Basket-AR

Basket-AR is a system for the automatic recognition of the main actions occurring in amateur basketball game videos.

The pipeline combines:

* **RF-DETR Large** for ball and basket detection;
* **MoViNet A2** for visual and temporal feature extraction;
* a **bidirectional GRU** for modeling geometric features;
* a **multi-scale sliding window** strategy for analyzing full-length videos;
* a **post-processing** stage for aggregating predictions and generating a temporal event report.

## Recognized actions

The model distinguishes the following classes:

* non-game;
* pass;
* two-point shot;
* three-point shot;
* free throw.

For shooting actions, the system also predicts the outcome, distinguishing between **made** and **missed** shots.

## Dataset

The dataset was created from 6 amateur basketball game videos recorded at 30 fps and manually annotated.

A second dataset was created to train the detector, containing:

* 1,321 annotated frames;
* 1,252 ball annotations;
* 1,176 basket annotations.

The training, validation, and test split was performed by source video in order to prevent data leakage.

## Main results

On the test set, the final model achieved:

| Metric                    |  Value |
| ------------------------- | -----: |
| Action accuracy           | 0.8477 |
| Macro precision           | 0.8046 |
| Macro recall              | 0.8248 |
| Macro F1-score            | 0.8113 |
| Outcome accuracy          | 0.8562 |
| Outcome balanced accuracy | 0.8500 |

## Main files

* `roboflow_Large.py`: RF-DETR training;
* `video_preprocessor.py`: clip preprocessing;
* `dataset.py`: dataset loading and management;
* `trainingMoViNet.py`: multimodal model training;
* inference scripts: full-video analysis and CSV report generation.


## Documentation

The full description of the architecture, dataset construction, experiments, and results is available in the report:

[Read the full Basket-AR report](./Basket_AR.pdf)

## Notes

The datasets, videos, and checkpoints may not be included in the repository because of their size and possible confidentiality constraints.

## Authors

Michele Quaglia, Salvatore Alberti
