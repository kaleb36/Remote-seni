Deep Hashing Models for Image Retrieval on Remote Sensing Datasets

This repository provides the implementation of four state-of-the-art deep hashing models for image retrieval,
specifically designed for three remote sensing datasets.


## Features
- Implementation of **four state-of-the-art deep hashing models**.
- Support for **three remote sensing datasets**:
  - **UC Merced Land Use Dataset (UCMD)**
  - **WHU-RS Dataset**
  - **Aerial Image Dataset (AID)**
- Comprehensive evaluation of hashing-based image retrieval performance.
- Customizable parameters for training and testing.

## Evaluation Results
###
| Model       | UCMD  | WHU-RS | AID   |
|-------------|-------|--------|-------|
| DPSH        | 0.9005| 0.8995 | 0.78  |
| IDHN        | 0.83  | 0.81   | 0.79  |
| GreedyHash  | 0.88  | 0.84   | 0.82  |
| HashNet     | 0.87  | 0.83   | 0.81  |

![WHURS results](fig/Picture1.png)
![UCMD results](fig/Picture2.png)
![AID results](fig/Picture3.png)

