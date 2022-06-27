import os
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = "../data"


def load_data():
    df_answers = pd.read_json(os.path.join(DATA_PATH, "inputs", "answers.json"), lines=True)
    df_questions = pd.read_json(os.path.join(DATA_PATH, "inputs", "questions.json"), lines=True)
    df_users = pd.read_json(os.path.join(DATA_PATH, "inputs", "users.json"), lines=True)
    print(f"df_answers.shape: {df_answers.shape}\n"
          f"df_questions.shape: {df_questions.shape}\n"
          f"df_users.shape: {df_users.shape}"
    )
    return df_answers, df_questions, df_users


def split_questions(df_questions, df_answers):
    df_top_questions = filter_top_questions(df_questions, df_answers)
    Q_train, Q_test = train_test_split(df_top_questions, test_size=1000)
    Q_train, Q_val = train_test_split(Q_train, test_size=5000)
    print(f"Q_train.shape: {Q_train.shape}\n"
          f"Q_val.shape: {Q_val.shape}\n"
          f"Q_test.shape: {Q_test.shape}"
    )
    return Q_train, Q_val, Q_test


def filter_top_questions(df_questions, df_answers, k_answer=1):
    count_question_answers = df_answers.groupby("question_id").answer_id.count()
    df_questions["n_answers"] = df_questions.question_id.map(count_question_answers)

    return df_questions.loc[df_questions.n_answers >= k_answer].reset_index()


def save_results_csv(file_name, content):
    cols = ["question_id", *[f"user{idx}_id" for idx in range(1, 21)]]
    file_path = os.path.join(DATA_PATH, "results", file_name)
    pd.DataFrame(content, columns=cols).to_csv(file_path, index=False)
    print(f"{file_path} written")