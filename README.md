# Deep Learning Project – WikiArt Classification

This project trains a deep learning model to classify paintings by artist using a subset of the WikiArt dataset.

## Repository Structure

deep_learning_project
├─ data
│  └─ .gitkeep
├─ notebooks
│  └─ project_notebook.ipynb
├─ scripts
│  └─ download_dataset.py
├─ src
├─ models
├─ README.md
└─ .gitignore

The dataset is not stored in the repository because it is too large.

## Setup

1. Clone the repository

git clone https://github.com/USERNAME/deep_learning_project.git
cd deep_learning_project

2. Install dependencies

pip install -r requirements.txt

3. Download the dataset

python scripts/download_dataset.py

After downloading, the structure should look like:

data/
   wikiart/
      Claude_Monet/
      Pablo_Picasso/
      Vincent_van_Gogh/
      ...

## Running the Notebook

Open the notebook and run the cells.

Example dataset path used in the code:

dataset_path = "data/wikiart"

## Dataset

The dataset is a subset of WikiArt containing paintings grouped by artist.  
Each artist folder represents one class in the classification task.

## Objective

Train and evaluate a deep learning model (using Keras) capable of classifying paintings by artist.