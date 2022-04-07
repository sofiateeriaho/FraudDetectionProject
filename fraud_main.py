# Author: Sofia Teeriaho
# Finishing date: 07.03.22
# See README for more information about running the code

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import tensorflow as tf
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, classification_report
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression

# Function to check for missing data
def checkMissing(data):

    print(data.isnull().sum().max(), 'missing values in dataset')

    # Drop missing data
    return data.dropna()

class DisplayPlots():
    def __init__(self, data, name):
        self.data = data
        self.name = name
        self.real = data[data["Class"] == 0]
        self.fraud = data[data["Class"] == 1]

    # Display ratio of fraud to non-fraud cases
    def PiePlotRatio(self):
        labels = ['Real', 'Fraud']
        plt.pie([len(self.real), len(self.fraud)], labels=labels, autopct='%.4f%%')
        plt.title("Ratio of Fraudulent Transactions")
        plt.savefig(self.name + "Ratio.png")
        plt.show()

    # Display non-PCA features' distribution plot
    def DistributionPlot(self, feature, type):

        fig, ax = plt.subplots(1, 2, figsize=(18, 4))

        real_values = self.real[feature].values
        fraud_values = self.fraud[feature].values

        if type == "Histogram":
            sns.histplot(real_values, ax=ax[0], color='b', bins=50)
            sns.histplot(fraud_values, ax=ax[1], color='r', bins=50)
            ax[0].set_ylabel("Amount of Transactions")
            ax[1].set_ylabel("Amount of Transactions")
        else:
            sns.distplot(real_values, ax=ax[0], color='b')
            sns.distplot(fraud_values, ax=ax[1], color='r')

        ax[0].set_title('Distribution of Non-fraudulent Transaction ' + feature, fontsize=14)
        ax[0].set_xlim([min(real_values), max(real_values)])
        ax[0].set_xlabel(feature)

        ax[1].set_title('Distribution of Fraudulent Transaction ' + feature, fontsize=14)
        ax[1].set_xlim([min(fraud_values), max(fraud_values)])
        ax[1].set_xlabel(feature)

        plt.savefig(self.name + feature + "Distribution.png")
        plt.show()

    # Display PCA features' distribution plot
    def PCADistributionPlots(self):

        features = self.data.columns[1:29].values
        print(features)

        fig, axes = plt.subplots(7, 4, figsize=(18, 20))
        plt.subplots_adjust(left=0.05,
                            bottom=0.05,
                            right=0.95,
                            top=0.95,
                            wspace=0.35,
                            hspace=0.35)

        # Iterate through PCA columns V1-V28
        for f, ax in zip(features, axes.ravel()):
            real_values = self.real[f].values
            fraud_values = self.fraud[f].values
            sns.distplot(real_values, ax=ax, color='b')
            sns.distplot(fraud_values, ax=ax, color='r')
            ax.set_title(f)
        plt.savefig("PCA_Distributions.png")
        plt.show()

# Class to display result metrics
class DisplayResults():
    def __init__(self, y_real, y_pred, method):
        self.y_real = y_real
        self.y_pred = y_pred
        self.method = method

    def evalMetrics(self):

        print('-----------Confusion Matrix for ' + self.method + '----')
        print(confusion_matrix(self.y_real, self.y_pred))

        print('----Binary Classification Metrics for '+ self.method + '----')
        print('Accuracy:', (accuracy_score(self.y_real, self.y_pred)))
        print('Recall:', (recall_score(self.y_real, self.y_pred)))
        print('Precision:', (precision_score(self.y_real, self.y_pred)))
        print('F1 score:', (f1_score(self.y_real, self.y_pred)))

    def rocCurve(self, y_scores):

        # Returns the false positive rate, true pos. rate, threshold
        fpr, tpr, threshold = roc_curve(self.y_real, y_scores[:, 1])

        # Calculate AUC (area under curve)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.title("ROC Curve for " + self.method)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig("ROC_" + self.method + ".png")
        plt.show()


# Function to randomly pick same amount of non-fraudulent cases as fraudulent cases to have an equal distribution of data
def randomUndersample(data):

    print("Undersampling data...")

    fraud = data[data["Class"] == 1]

    sample_list = []
    for i in range(0, len(data)):
        if data.iloc[i]['Class'] == 0:
           sample_list.append(i)

    random.seed(100)

    # Select random index values
    sampleIDX = random.sample(sample_list, len(sample_list) - len(fraud))

    return data.drop(sampleIDX)

# Split data into training and testing sets
def splitData(data):

    # x data independent of class while y only contains class label
    x = data.drop(["Class"], axis=1)
    y = data["Class"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test

# Decision Tree classification model using scikit-learn library
def decisionTreeModel(x_train, x_test, y_train):

    tree_model = DecisionTreeClassifier(max_depth=5, criterion='entropy')
    tree_model.fit(x_train, y_train)

    # Default threshold of 0.5
    tree_pred = tree_model.predict(x_test)

    y_scores = tree_model.predict_proba(x_test)

    # Set own threshold from ROC curve (uncomment for results)
    #tree_pred = [1 if prob > 0.085 else 0 for prob in np.ravel(y_scores[:, 1])]

    return tree_pred, y_scores

# K-nearest neighbor classification model using scikit-learn library
def kNearestNeighborModel(x_train, x_test, y_train):

    knn = KNN(n_neighbors=5)

    # train the model
    knn.fit(x_train, y_train)

    # Default threshold of 0.5
    y_pred = knn.predict(x_test)

    y_scores = knn.predict_proba(x_test)

    # Set own threshold from ROC curve (uncomment for results)
    #y_pred = [1 if prob > 0.2 else 0 for prob in np.ravel(y_scores[:, 1])]

    return y_pred, y_scores

# Logistic Regression classification model using scikit-learn library
def logisticRegression(x_train, x_test, y_train):

    model = LogisticRegression(class_weight='balanced')
    #model.fit(x_train.reshape(x_train.shape[0], 1), y_train)
    model.fit(x_train, y_train)

    # Default threshold of 0.5
    y_pred = model.predict(x_test)

    y_scores = model.predict_proba(x_test)

    # Set own threshold from ROC curve (uncomment for results)
    #y_pred = [1 if prob > 0.8 else 0 for prob in np.ravel(y_scores[:, 1])]

    return y_pred, y_scores

# Neural Network using Tensorflow library
def deepLearning(x_train, x_test, y_train):

    # Scale data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    tf.random.set_seed(42)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(lr=0.03),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    model.fit(x_train_scaled, y_train, epochs=100)

    model.predict(x_test_scaled)

# Drop features that were highly correlated in PCA distribution plots
def cleanData(data):

    return data.drop(["V13", "V15", "V22", "V25"], axis=1)

# Display all the models' evaluation metrics in order for easy comparison
def displayAllEvalMetrics(x_train, x_test, y_train, y_test):

    y_pred, y_scores = decisionTreeModel(x_train, x_test, y_train)
    results = DisplayResults(y_test, y_pred, "Tree")
    results.evalMetrics()

    y_pred, y_scores = kNearestNeighborModel(x_train, x_test, y_train)
    results = DisplayResults(y_test, y_pred, "KNN")
    results.evalMetrics()

    y_pred, y_scores = logisticRegression(x_train, x_test, y_train)
    results = DisplayResults(y_test, y_pred, "Logistic Regression")
    results.evalMetrics()

    deepLearning(x_train, x_test, y_train)

def askUserforMethod(x_train, x_test, y_train, y_test):

    print("\nWhich prediction method would you like to see\n")
    print("1. Decision Tree \n2. K-Nearest Neighbor \n3. Logistic Regression \n4. Neural Networks")
    choice = input("Which prediction method would you like to see: ")
    print("Processing data...")

    y_pred = 0
    method = " "

    if choice == '1':
        print("Building tree...")
        y_pred, y_scores = decisionTreeModel(x_train, x_test, y_train)
        method = "Tree"

        results = DisplayResults(y_test, y_pred, method)
        results.evalMetrics()

        # Uncomment to see ROC curves
        #results.rocCurve(y_scores)

    elif choice == '2':
        print("Identifying neighbors...")
        y_pred, y_scores = kNearestNeighborModel(x_train, x_test, y_train)
        method = "KNN"

        results = DisplayResults(y_test, y_pred, method)
        results.evalMetrics()

        # Uncomment to see ROC curves
        #results.rocCurve(y_scores)

    elif choice == '3':
        print("...")
        y_pred, y_scores = logisticRegression(x_train, x_test, y_train)
        method = "Logistic Regression"

        results = DisplayResults(y_test, y_pred, method)
        results.evalMetrics()

        # Uncomment to see ROC curves
        #results.rocCurve(y_scores)

    elif choice == '4':
        print("Firing neurons...")
        deepLearning(x_train, x_test, y_train)
        method = "Neural Network"
    else:
        print("Nothing chosen")


def run_main():

    # Uncomment to display raw data plots
    #raw = DisplayPlots(df, "Raw")
    #raw.DistributionPlot("Amount", "Dis")
    #raw.DistributionPlot("Time", "Histogram")
    #raw.PCADistributionPlots()
    #raw.CorrelationHeatMap()

    print("Importing data...")
    df = pd.read_csv('creditcard.csv')

    # Undersample data
    data = randomUndersample(df)

    # Uncomment to display equal distribution data plots
    #equal = DisplayPlots(data, "UnderSampled")
    #equal.DistributionPlot("Amount", "Dis")
    #equal.DistributionPlot("Time", "Histogram")
    #equal.PCADistributionPlots()

    # Normalize data with correlated features from PCA distribution plot (uncomment for results)
    # ata = cleanData(data)

    # Split data
    x_train, x_test, y_train, y_test = splitData(data)

    # Uncomment to display all classification model results at once
    #displayAllEvalMetrics(x_train, x_test, y_train, y_test)

    askUserforMethod(x_train, x_test, y_train, y_test)

if __name__ == '__main__':
    run_main()

