# create training script (top line has to be the first line in a cell)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import sklearn
import joblib
import boto3
import pathlib
from io import StringIO 
import argparse
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
   
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "k_model.joblib"))
    return clf        
    
if __name__ == "__main__":

    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="oulad_train.csv")
    parser.add_argument("--test-file", type=str, default="oulad_test.csv")

    args, _ = parser.parse_known_args()
    
    print("SKLearn Version: ", sklearn.__version__)
    print("Joblib Version: ", joblib.__version__)

    print("[INFO] Reading data")
    print()
   
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))
    
    features = list(train_df.columns)
    label = "final_result"
    
    print("Building training and testing datasets")
    print()
    
    # change  "total_score*weight"  because target column has to be numerical
    
    X_train = train_df.drop( label, axis=1)
    y_train = train_df[label]
    X_test = test_df.drop(label, axis=1)
    y_test = test_df[label]

    print('Column order: ')
    print(features)
    print()
    
    print("Label column is: ",label)
    print()
    
    print("Data Shape: ")
    print()
    print("---- SHAPE OF TRAINING DATA (70%) ----")
    print(X_train.shape)
    print(y_train.shape)
    print()
    print("---- SHAPE OF TESTING DATA (30%) ----")
    print(X_test.shape)
    print(y_test.shape)
    print()
    
  
    print("Training K-means Model.....")
    print()
    
    # Perform Simple K-means clustering with k=3 clusters
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_train)
    print("kmeans---------")
    print(kmeans)
    # Add cluster labels to original dataframe
    X_train['Cluster'] = kmeans.labels_
    print("X_train with cluster---------")
    print(X_train)

    # Perform association rule mining on clustered data
    frequent_itemsets = apriori(X_train.groupby('Cluster')['highest_education'], min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

    # Print resulting association rules
    print("rules---------")
    print(rules)

    model = rules
    model_path = os.path.join(args.model_dir, "k_model.joblib")
    joblib.dump(model,model_path)
    print("Model persisted at " + model_path)
    print()

    
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test,y_pred_test)
    test_rep = classification_report(y_test,y_pred_test)

    print()
    print("---- METRICS RESULTS FOR TESTING DATA ----")
    print()
    print("Total Rows are: ", X_test.shape[0])
    print('[TESTING] Model Accuracy is: ', test_acc)
    print('[TESTING] Testing Report: ')
    print(test_rep)
    
    # result Visualisation
