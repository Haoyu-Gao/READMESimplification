import pandas as pd
import numpy as np

import pickle
from scipy import spatial
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from matplotlib import pyplot as plt


vectorizer = pickle.load(open("../data/tfidf.pkl", "rb"))


def calculation(threshold):
    data = pd.read_csv(f"../data/labeled_alignment.csv", header=None)
    data.columns = ['idx', 'simp_idx', 'norm_idx', 'simp_sent', 'norm_sent', 'aligned']
    predicted = []
    ground = []
    for i in range(len(data)):
        simp, norm, label = data.iloc[i, 3], data.iloc[i, 4], data.iloc[i, 5]
        tfidf = vectorizer.transform([simp, norm]).todense()
        print(tfidf[0, :].shape)
        d = spatial.distance.cosine(tfidf[0, :], tfidf[1, :])
        if d > threshold:
            predicted.append(0)
        else:
            predicted.append(1)
        if label.strip() == "False":
            ground.append(0)
        else:
            ground.append(1)

    accuracy = accuracy_score(ground, predicted)
    recall = recall_score(ground, predicted)
    precision = precision_score(ground, predicted)
    f1 = f1_score(ground, predicted)

    print("Accuracy score = {}".format(accuracy))
    print("Recall score = {}".format(recall))
    print("Precision score = {}".format(precision))
    print("F1 score = {}".format(f1))

    return accuracy, recall, precision, f1


if __name__ == '__main__':
    thresholds = np.arange(0.05, 0.85, 0.05)
    accuracies, recalls, precisions, f1s = [], [], [], []

    for i in thresholds:
        a, r, p, f = calculation(i)
        accuracies.append(a)
        recalls.append(r)
        precisions.append(p)
        f1s.append(f)

    plt.plot(thresholds, accuracies, label='accuracy')
    plt.plot(thresholds, recalls, label='recall')
    plt.plot(thresholds, precisions, label='precision')
    plt.plot(thresholds, f1s, label='f1')
    plt.legend()
    plt.xlabel('threshold')
    
    plt.show()
