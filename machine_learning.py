# Step 1 import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# # Step 2 read the csv files and create pandas dataframes
legitimate_df = pd.read_csv("structured_data_legitimate1.csv")
phishing_df = pd.read_csv("structured_data_phishing1.csv")

# Step 3 combine legitimate and phishing dataframes, and shuffle
df1 = pd.concat([legitimate_df, phishing_df], axis=0)
df = df1.dropna()

df = df.sample(frac=1)

# Move the URL and label columns to the last
df = df[[col for col in df if col not in ['URL', 'label']] + ['URL', 'label']]

# newdf=pd.read_csv("url_content_combined_dataset.csv", index=False)
print(df.head())

# Save DataFrame to CSV file
# df.to_csv('url_content_combined_dataset.csv', index=False)

# Step 4 remove'url' and remove duplicates, then we can create X and Y for the models, Supervised Learning
df = df.drop('URL', axis=1)

df = df.drop_duplicates()

X = df.drop('label', axis=1)
Y = df['label']

# Step 5 split data to train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=60)

# Decision Tree
dt_model = tree.DecisionTreeClassifier()

# Gaussian Naive Bayes
nb_model = GaussianNB()

# AdaBoost
ada_model = AdaBoostClassifier()

# XGBoost
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

# K-fold cross validation, and K = 5
K = 5
total = X.shape[0]
index = int(total / K)

# 1
X_1_test = X.iloc[:index]
X_1_train = X.iloc[index:]
Y_1_test = Y.iloc[:index]
Y_1_train = Y.iloc[index:]

# 2
X_2_test = X.iloc[index:index * 2]
X_2_train = X.iloc[np.r_[:index, index * 2:]]
Y_2_test = Y.iloc[index:index * 2]
Y_2_train = Y.iloc[np.r_[:index, index * 2:]]

# 3
X_3_test = X.iloc[index * 2:index * 3]
X_3_train = X.iloc[np.r_[:index * 2, index * 3:]]
Y_3_test = Y.iloc[index * 2:index * 3]
Y_3_train = Y.iloc[np.r_[:index * 2, index * 3:]]

# 4
X_4_test = X.iloc[index * 3:index * 4]
X_4_train = X.iloc[np.r_[:index * 3, index * 4:]]
Y_4_test = Y.iloc[index * 3:index * 4]
Y_4_train = Y.iloc[np.r_[:index * 3, index * 4:]]

# 5
X_5_test = X.iloc[index * 4:]
X_5_train = X.iloc[:index * 4]
Y_5_test = Y.iloc[index * 4:]
Y_5_train = Y.iloc[:index * 4]

# X and Y train and test lists
X_train_list = [X_1_train, X_2_train, X_3_train, X_4_train, X_5_train]
X_test_list = [X_1_test, X_2_test, X_3_test, X_4_test, X_5_test]

Y_train_list = [Y_1_train, Y_2_train, Y_3_train, Y_4_train, Y_5_train]
Y_test_list = [Y_1_test, Y_2_test, Y_3_test, Y_4_test, Y_5_test]


def calculate_measures(TN, TP, FN, FP):
    model_accuracy = (TP + TN) / (TP + TN + FN + FP)
    model_precision = TP / (TP + FP)
    model_recall = TP / (TP + FN)
    return model_accuracy, model_precision, model_recall
#
#
rf_accuracy_list, rf_precision_list, rf_recall_list = [], [], []
dt_accuracy_list, dt_precision_list, dt_recall_list = [], [], []
nbayes_accuracy_list, nbayes_precision_list, nbayes_recall_list = [], [], []
ada_accuracy_list, ada_precision_list, ada_recall_list = [], [], []
xgb_accuracy_list, xgb_precision_list, xgb_recall_list = [], [], []

for i in range(0, K):
     # ----- RANDOM FOREST ----- #
    rf_model.fit(X_train_list[i], Y_train_list[i])
    rf_predictions = rf_model.predict(X_test_list[i])
#     tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=rf_predictions).ravel()
#     rf_accuracy, rf_precision, rf_recall = calculate_measures(tn, tp, fn, fp)
#     rf_accuracy_list.append(rf_accuracy)
#     rf_precision_list.append(rf_precision)
#     rf_recall_list.append(rf_recall)
#
#     # ----- DECISION TREE ----- #
    dt_model.fit(X_train_list[i], Y_train_list[i])
    dt_predictions = dt_model.predict(X_test_list[i])
#     tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=dt_predictions).ravel()
#     dt_accuracy, dt_precision, dt_recall = calculate_measures(tn, tp, fn, fp)
#     dt_accuracy_list.append(dt_accuracy)
#     dt_precision_list.append(dt_precision)
#     dt_recall_list.append(dt_recall)
#
#     # ----- NAIVE BAYES ----- #
    nb_model.fit(X_train_list[i], Y_train_list[i])
    nbayes_predictions = nb_model.predict(X_test_list[i])
#     tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=nbayes_predictions).ravel()
#     nbayes_accuracy, nbayes_precision, nbayes_recall = calculate_measures(tn, tp, fn, fp)
#     nbayes_accuracy_list.append(nbayes_accuracy)
#     nbayes_precision_list.append(nbayes_precision)
#     nbayes_recall_list.append(nbayes_recall)
#     #
#     # ----- ADABOOST ----- #
    ada_model.fit(X_train_list[i], Y_train_list[i])
    ada_predictions = ada_model.predict(X_test_list[i])
#     tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=ada_predictions).ravel()
#     ada_accuracy, ada_precision, ada_recall = calculate_measures(tn, tp, fn, fp)
#     ada_accuracy_list.append(ada_accuracy)
#     ada_precision_list.append(ada_precision)
#     ada_recall_list.append(ada_recall)
#
#     # ----- XGBOOST ----- #
    xgb_model.fit(X_train_list[i], Y_train_list[i])
    xgb_predictions = xgb_model.predict(X_test_list[i])
#     tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=xgb_predictions).ravel()
#     xgb_accuracy, xgb_precision, xgb_recall = calculate_measures(tn, tp, fn, fp)
#     xgb_accuracy_list.append(xgb_accuracy)
#     xgb_precision_list.append(xgb_precision)
#     xgb_recall_list.append(xgb_recall)
#
# # Random Forest
# RF_accuracy = sum(rf_accuracy_list) / len(rf_accuracy_list)
# RF_precision = sum(rf_precision_list) / len(rf_precision_list)
# RF_recall = sum(rf_recall_list) / len(rf_recall_list)
#
# print("Random Forest accuracy ==> ", RF_accuracy)
# print("Random Forest precision ==> ", RF_precision)
# print("Random Forest recall ==> ", RF_recall)
#
# # Decision Tree
# DT_accuracy = sum(dt_accuracy_list) / len(dt_accuracy_list)
# DT_precision = sum(dt_precision_list) / len(dt_precision_list)
# DT_recall = sum(dt_recall_list) / len(dt_recall_list)
#
# print("Decision Tree accuracy ==> ", DT_accuracy)
# print("Decision Tree precision ==> ", DT_precision)
# print("Decision Tree recall ==> ", DT_recall)
#
# # Naive Bayes
# NB_accuracy = sum(nbayes_accuracy_list) / len(nbayes_accuracy_list)
# NB_precision = sum(nbayes_precision_list) / len(nbayes_precision_list)
# NB_recall = sum(nbayes_recall_list) / len(nbayes_recall_list)
#
# print("Naive Bayes accuracy ==> ", NB_accuracy)
# print("Naive Bayes precision ==> ", NB_precision)
# print("Naive Bayes recall ==> ", NB_recall)
#
# # AdaBoost
# ADA_accuracy = sum(ada_accuracy_list) / len(ada_accuracy_list)
# ADA_precision = sum(ada_precision_list) / len(ada_precision_list)
# ADA_recall = sum(ada_recall_list) / len(ada_recall_list)
#
# print("AdaBoost accuracy ==> ", ADA_accuracy)
# print("AdaBoost precision ==> ", ADA_precision)
# print("AdaBoost recall ==> ", ADA_recall)
#
# # XGBoost
# XGB_accuracy = sum(xgb_accuracy_list) / len(xgb_accuracy_list)
# XGB_precision = sum(xgb_precision_list) / len(xgb_precision_list)
# XGB_recall = sum(xgb_recall_list) / len(xgb_recall_list)
#
# print("XGBoost accuracy ==> ", XGB_accuracy)
# print("XGBoost precision ==> ", XGB_precision)
# print("XGBoost recall ==> ", XGB_recall)

# data = {'accuracy': [RF_accuracy, DT_accuracy, NB_accuracy, ADA_accuracy, RF_accuracy],
#         'precision': [RF_precision, DT_precision, NB_precision, ADA_precision, XGB_precision],
#         'recall': [RF_recall, DT_recall, NB_recall, ADA_recall, XGB_recall]
#         }

index = ['RF', 'DT', 'NB', 'ADA', 'XGB']

# df_results = pd.DataFrame(data=data, index=index)
#
# # visualize the dataframe
# ax = df_results.plot.bar(rot=0)
# plt.show()


#
# # Random Forest
# rf_model = RandomForestClassifier(n_estimators=60)
# rf_model.fit(x_train, y_train)
# rf_predictions = rf_model.predict(x_test)
# rf_accuracy = accuracy_score(y_test, rf_predictions)
# rf_precision = precision_score(y_test, rf_predictions)
# rf_recall = recall_score(y_test, rf_predictions)
#
# # Decision Tree
# dt_model = tree.DecisionTreeClassifier()
# dt_model.fit(x_train, y_train)
# dt_predictions = dt_model.predict(x_test)
# dt_accuracy = accuracy_score(y_test, dt_predictions)
# dt_precision = precision_score(y_test, dt_predictions)
# dt_recall = recall_score(y_test, dt_predictions)
#
# # Gaussian Naive Bayes
# nb_model = GaussianNB()
# nb_model.fit(x_train, y_train)
# nb_predictions = nb_model.predict(x_test)
# nb_accuracy = accuracy_score(y_test, nb_predictions)
# nb_precision = precision_score(y_test, nb_predictions)
# nb_recall = recall_score(y_test, nb_predictions)
#
# # AdaBoost
# ada_model = AdaBoostClassifier()
# ada_model.fit(x_train, y_train)
# ada_predictions = ada_model.predict(x_test)
# ada_accuracy = accuracy_score(y_test, ada_predictions)
# ada_precision = precision_score(y_test, ada_predictions)
# ada_recall = recall_score(y_test, ada_predictions)
#
# # XGBoost
# xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
# xgb_model.fit(x_train, y_train)
# xgb_predictions = xgb_model.predict(x_test)
# xgb_accuracy = accuracy_score(y_test, xgb_predictions)
# xgb_precision = precision_score(y_test, xgb_predictions)
# xgb_recall = recall_score(y_test, xgb_predictions)
#
# # Plotting
# models = ['RF', 'DT', 'NBayes', 'Ada', 'XGB']
# accuracies = [rf_accuracy * 100, dt_accuracy * 100, nb_accuracy * 100, ada_accuracy * 100, xgb_accuracy * 100]
# precisions = [rf_precision * 100, dt_precision * 100, nb_precision * 100, ada_precision * 100, xgb_precision * 100]
# recalls = [rf_recall * 100, dt_recall * 100, nb_recall * 100, ada_recall * 100, xgb_recall * 100]
#
# x = np.arange(len(models))  # the label locations
# width = 0.25  # the width of the bars
#
# fig, ax = plt.subplots(figsize=(10, 6))
#
# rects1 = ax.bar(x - width, accuracies, width, label='Accuracy', color='skyblue')
# rects2 = ax.bar(x, precisions, width, label='Precision', color='salmon')
# rects3 = ax.bar(x + width, recalls, width, label='Recall', color='lightgreen')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xlabel('Models')
# ax.set_ylabel('Scores (%)')
# ax.set_title('Model Performance Metrics')
# ax.set_xticks(x)
# ax.set_xticklabels(models)
# ax.legend()

#
# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(round(height, 2)),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
#
# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
#
# plt.show()

models = ['RF', 'DT', 'Ada', 'NBayes', 'XGB']
accuracies = [85, 78, 85, 58, 90]
new_accuracies = [89, 80, 90, 61, 93]
colors = ['#cfb148', '#cc4949', '#8c679e', '#5d8f8e', '#632424']

# Create subplots with 1 row and 2 columns
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the first bar plot
bars1 = axs[0].bar(models, accuracies, color=colors)
axs[0].set_xlabel('Models')
axs[0].set_ylabel('Accuracy')
axs[0].set_title('Accuracy of Different Models')
axs[0].set_ylim(0, 100)  # Set the y-axis limit to ensure the accuracy values are displayed properly

# Plot the second bar plot
bars2 = axs[1].bar(models, new_accuracies, color=colors)
axs[1].set_xlabel('Models')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Accuracy of Different Models (New)')
axs[1].set_ylim(0, 100)  # Set the y-axis limit to ensure the accuracy values are displayed properly


# Function to add value labels on top of bars
def add_labels(bars, ax):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# Add value labels on top of bars for both plots
add_labels(bars1, axs[0])
add_labels(bars2, axs[1])

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()
