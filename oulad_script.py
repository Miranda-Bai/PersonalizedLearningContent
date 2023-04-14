# create training script (top line has to be the first line in a cell)

from sklearn.ensemble import RandomForestClassifier
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

# inference functions ---------------

# def input_fn(request_body, request_content_type):
#     print(request_body)
#     print(request_content_type)
#     if request_content_type == "text/csv":
#         request_body = request_body.strip()
#         try:
#             df = pd.read_csv(StringIO(request_body), header=None)
#             return df
        
#         except Exception as e:
#             print(e)
#     else:
#         return """Please use Content-Type = 'text/csv' and, send the request!!""" 
 
    
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

# def predict_fn(input_data, model):
#     if type(input_data) != str:
#         prediction = model.predict(input_data)
#         print(prediction)
#         return prediction
#     else:
#         return input_data
        
    
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
    
  
    print("Training RandomForest Model.....")
    print()
    model =  RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, verbose = 3,n_jobs=-1)
    # print("model is --------")
    # print(model)
    # print("X_train, y_train:----------")
    # print(X_train)
    # print(y_train)
    model.fit(X_train, y_train)
    print()
    

    model_path = os.path.join(args.model_dir, "model.joblib")
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
    
    # Tree Visualisation
from sklearn.tree import export_graphviz
import graphviz
# from sklearn import tree
# import matplotlib.pyplot as plt

# Export the first three decision trees from the forest
rf = model
cn=['Withdrawn', 'Fail', 'Pass', 'Distinction']
for i in range(3):
    tree = rf.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=X_train.columns,  
                               class_names=cn,
                               filled=True,  
                               max_depth=2, 
                               impurity=False, 
                               proportion=True)
    graph = graphviz.Source(dot_data)
    print("******graph******")
    print(graph)
    # Image(data=graph)
    # display(graph) # can't use it
