# Algolia ML Challenge Solution

## Solution

This challenge proposes to predict the 20 most likely Stack Overflow users to answer a question. In my understanding, our goal is to predict users for new questions, not suggesting users for questions that have been asked already.

In that sense, we will split the dataset into a train, validation and test dataset. The train set will be used to fit models, while we will run our users predictions for questions from the validation set. The test set will be left untouched to evaluate a final model. 

This setting prevent us from using collaborative filtering method, because a new question wouldn't have answers nor users associated to it. This question would face the cold start problem, like a new movie in the netflix catalog.

Instead, we will leverage content-based approaches, by focusing on questions title and description. The core idea is to use different embedding methods (BERT embeddings and later a simpler TF-IDF embeddings) to generate a candidate pools of nearest neighbors for a new question. Then, we will fetch users having answered to these questions and rank them using their answer ratings and users attributes.

## Content

### `notebooks`

This repository contains the notebooks of different approaches to solve the challenge, along with an introductory exploration data analysis (EDA). The notebooks content is organised as follow:
1. EDA
2. Baseline 1: Question and Users Embeddings ⚠️ without using precompute embeddings, the notebook will take long to execute (~1h30)
3. Baseline 2: ANN & Candidate Users Ranking ⚠️ without using precompute embeddings, the notebook will take long to execute (~1h)
4. Baseline 3. TF-IDF Embeddings & Candidate Users Ranking 🟢 fast embeddings

### `data`

The data folder contains the following subfolders:
- `inputs` for raw json inputs that you need to download and unzip from the [original challenge repository](https://github.com/algolia/ML-challenge).
- `intermediary` for precompute embeddings that you can optionally download from [my google drive](https://drive.google.com/drive/folders/1H2vRzdrCJNiuiSxXDHbQGe-aYkuriC3r?usp=sharing) for a significant compute speed-up.
- `results` for results of different approaches.

### `src`

Various utils to reduce the amount of code from notebook and offer consistent methods. It also contains the BERT embedder model used accross different approaches.


## Setup

### Requirements
- python 3.7 and higher

### Launching a notebook

Download this repository
```shell
git clone git@github.com:Vincent-Maladiere/Algolia-ML-challenge-solution.git
```

Create a python environment and source it
```shell
python -m venv venv && source venv/bin/activate
```

Create data folders
```
mkdir -p data/inputs data/intermediary data/results
```

Download [input data](https://drive.google.com/file/d/1CUcfl3JX8TNYABn2JRIPQozT0oqdqqOy/view) and place the zip file in `data/inputs`, then unzip it:
on OSX, run
```
cd data/inputs && unzip ml_challenge.zip && cd ../..
```

[Optional] download [embeddings precompute data](https://drive.google.com/drive/folders/1H2vRzdrCJNiuiSxXDHbQGe-aYkuriC3r?usp=sharing) and place the 2 pickle files in `data/intermediary`


Install requirements
```
pip install -r requirements.txt
```

Finally, open a notebook by running:
```
jupyter notebook notebooks --log-level=CRITICAL&
```

