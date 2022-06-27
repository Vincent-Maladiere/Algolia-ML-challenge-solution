# Algolia ML Challenge Solution

## Solution

This challenge proposes to predict the 20 most likely Stack Overflow users to answer a question. In my understanding, our goal is to predict users for new questions, not suggesting users for questions that already has been asked.

In that sense, we will split the dataset into a train, validation and test dataset. The train set will be used to fit models, while we will run our users predictions for questions from the validation set. The test set will be left untouched to evaluate a final model. 

This setting prevent us from using collaborative filtering method, because a new question wouldn't have answers nor users associated to it. This question would face the cold start problem, like a new movie in the netflix catalog.

Instead, we will leverage content-based approaches, by focusing on questions title and description. The core idea is to use different embedding methods (BERT embeddings and later a simpler TF-IDF embeddings) to generate a candidate pools of nearest neighbors for a new question. Then, we will fetch users having answered to these questions and rank them using their answer ratings and users attributes.

## Content

### `notebooks`

This repository contains the notebooks of different approaches to solve the challenge, along with an introductory exploration data analysis (EDA). The notebooks content is organised as follow:
1. EDA
2. Baseline 1: Question and Users Embeddings
3. Baseline 2: ANN & Candidate Users Ranking
4. Baseline 3. TF-IDF Embeddings & Candidate Users Ranking

### `data`

The data folder contains the following subfolders:
- `inputs` for raw json inputs that you need to download and unzip from the [original challenge repository](https://github.com/algolia/ML-challenge).
- `intermediary` for precompute embeddings.
- `results` for results of different approaches.

### `src`

Various utils to reduce the amount of code from notebook and offer consistent methods. It also contains the BERT embedder model used accross different approaches.


## Setup

### Requirements
- python 3.7 and higher

### Launching a notebook

Download this repository
```shell
git@github.com:Vincent-Maladiere/Algolia-ML-challenge-solution.git
```

Create a python environment and source it
```shell
python -m venv venv && source venv/bin/activate
```

Create data folders
```
mkdir -p data/inputs data/intermediary data/results
```

Install requirements
```
pip install -r requirements.txt
```

Download data and place the zip file in `data/input`, then unzip it:
on OSX, run
```
cd data/inputs && unzip ml_challenge.zip && cd ../..
```

Finally, open a notebook by running:
```
jupyter notebook notebooks --log-level=CRITICAL&
```

