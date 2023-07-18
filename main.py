import pandas as pd
import time
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
import os, psutil


RAW_PATH  = 'data/output/raw/'
SILVER_PATH  = 'data/output/silver/'
GOLD_PATH  = 'data/output/gold/'
STROKE_FILE_NAME = 'stroke.parquet'

def process_raw():
    df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
    df.to_parquet(RAW_PATH + STROKE_FILE_NAME)

def process_silver():
    df = pd.read_parquet(RAW_PATH + STROKE_FILE_NAME)

    df['bmi'].fillna(df['bmi'].mean(), inplace=True)

    df.to_parquet(SILVER_PATH + STROKE_FILE_NAME)

def process_gold():
    lab = LabelEncoder()
    scaler = MinMaxScaler()
    df = pd.read_parquet(SILVER_PATH + STROKE_FILE_NAME)

    df['bmi'].fillna(df['bmi'].mean(), inplace=True)
    df['gender'] = lab.fit_transform(df['gender'])
    df['ever_married'] = lab.fit_transform(df['ever_married'])
    df['Residence_type'] = lab.fit_transform(df['Residence_type'])
    df['smoking_status'] = lab.fit_transform(df['smoking_status'])
    df['work_type'] = lab.fit_transform(df['work_type'])

    df[['age', 'avg_glucose_level', 'bmi']] = scaler.fit_transform(df[['age', 'avg_glucose_level', 'bmi']])

    normalized_df=(df-df.min())/(df.max()-df.min())


    normalized_df.to_parquet(GOLD_PATH + STROKE_FILE_NAME)
    # normalized_df.to_csv(GOLD_PATH + 'test.csv')

def train_dataset():
    df = pd.read_parquet(GOLD_PATH + STROKE_FILE_NAME)
    X = df.drop("stroke", axis=1)
    y = df.stroke
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, shuffle=True, test_size=0.3)

    imputer = KNNImputer(n_neighbors=2)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.fit_transform(X_test)
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)

    score = clf.score(X_test, y_test)

    scores = cross_val_score(clf, X, y, cv=3)

    CM = confusion_matrix(y_test, y_predicted)

    print(score)
    print(f"Mean test score: {scores.mean()}")
    print(scores)

    print("TP: ", CM[1][1])
    print("FP: ", CM[0][1])
    print("TN: ", CM[0][0])
    print("FN: ", CM[1][0])

def main():
    start_memory = psutil.Process(os.getpid()).memory_info().rss

    time_start = time.perf_counter()
    process_raw()
    
    process_silver_start = time.perf_counter()
    print ("process_raw() took %5.8f secs" % (process_silver_start - time_start))

    process_silver()

    process_gold_start = time.perf_counter()
    print ("process_silver() took %5.8f secs" % (process_gold_start - process_silver_start))

    process_gold()

    train_dataset_start = time.perf_counter()
    print ("process_gold() took %5.8f secs" % (train_dataset_start - process_gold_start))

    train_dataset()

    print ("train_dataset() took %5.8f secs" % (time.perf_counter() - train_dataset_start))

    time_elapsed = (time.perf_counter() - time_start)
    print ("Total program took: %5.8f secs" % (time_elapsed))

    end_memory = psutil.Process(os.getpid()).memory_info().rss
    print(f"Total memory used: {(end_memory - start_memory)  / 1024 ** 2 } MB")

main()