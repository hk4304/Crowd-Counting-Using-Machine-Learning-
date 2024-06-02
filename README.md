# Crowd-Counting-Using-Machine-Learning-

This repository, maintained by hk4304, contains code for processing the ShanghaiTech Crowd Counting Dataset. The primary goal is to load image and label data, organize it into a DataFrame, and prepare it for subsequent machine learning tasks.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Data Processing](#data-processing)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Crowd counting is a crucial task in various applications such as surveillance, event management, and urban planning. This project focuses on preparing the ShanghaiTech Crowd Counting Dataset for machine learning models. The dataset includes images and corresponding ground truth data indicating the number of people present.

## Dataset
The ShanghaiTech Crowd Counting Dataset consists of two parts:
- **Part A**: Images with a high density of people.
- **Part B**: Images with a lower density of people.

The dataset can be downloaded from the [official website](http://www.crowdcounting.com/). Ensure to extract the dataset into a directory that you can access in your environment.

## Installation
To run the code in this repository, you need to have Python installed along with the following libraries:
- numpy
- pandas
- matplotlib
- opencv-python
- scipy
- pillow

You can install the required libraries using pip:
```sh
pip install numpy pandas matplotlib opencv-python scipy pillow
```

## Usage
1. **Clone the repository:**
    ```sh
    git clone https://github.com/hk4304/Crowd-Counting-Using-Machine-Learning.git
    cd Crowd-Counting-Using-Machine-Learning
    ```

2. **Open the Jupyter Notebook:**
   Open `ML_NOV_GLCM_0001.ipynb` in Jupyter Notebook or Jupyter Lab to see the data loading and processing steps.

3. **Dataset Preparation:**
   Ensure that your dataset is organized in the following directory structure:
   ```
   /content/drive/MyDrive/SML_PROJECT/
   ├── ShanghaiTech_Crowd_Counting_Dataset/
       ├── part_A_final/
           ├── train_data/
               ├── images/
               ├── ground_truth/
       ├── part_B_final/
           ├── train_data/
               ├── images/
               ├── ground_truth/
   ```

4. **Run the notebook cells:**
   Execute the cells in the notebook to load the images and labels into a DataFrame. The notebook includes code to mount Google Drive, import necessary libraries, and create a DataFrame with paths to images and their corresponding label files.

## Data Processing
The notebook includes the following key steps:
1. **Mount Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. **Import libraries:**
   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import cv2
   import os
   import scipy.io as sio
   from scipy.io import loadmat
   from PIL import Image
   ```
3. **Set directories and create DataFrame:**
   ```python
   image_dirA = "/content/drive/MyDrive/SML_PROJECT/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/images"
   label_dirA = "/content/drive/MyDrive/SML_PROJECT/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/ground_truth"

   image_dirB = "/content/drive/MyDrive/SML_PROJECT/ShanghaiTech_Crowd_Counting_Dataset/part_B_final/train_data/images"
   label_dirB = "/content/drive/MyDrive/SML_PROJECT/ShanghaiTech_Crowd_Counting_Dataset/part_B_final/train_data/ground_truth"

   def create_df(image_dir, label_dir):
       image_files = os.listdir(image_dir)
       image_files.sort()
       label_files = os.listdir(label_dir)
       label_files.sort()

       # Create full paths for image and label files
       image_paths = [os.path.join(image_dir, img) for img in image_files]
       label_paths = [os.path.join(label_dir, lbl) for lbl in label_files]

       dataframe = pd.DataFrame({'image_path': image_paths, 'label_path': label_paths})
       return dataframe

   df = create_df(image_dirB, label_dirB)
   print(df.shape)
   ```

4. **Display sample data:**
   ```python
   print(df["image_path"][5])
   ```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes. Follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.
