import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load dataset
df = pd.read_csv('/muhammadaarif/IOT/path_mit_bih.csv')

# Split dataset into features and labels
X = df.drop('label', axis=1).values
y = df['label'].values

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
svm_clf_acc = svm_clf.score(X_test, y_test)
svm_clf_y_pred = svm_clf.predict(X_test)

print("SVM Classifier accuracy:", svm_clf_acc)

# Train SVM regression
svm_reg = SVR()
svm_reg.fit(X_train, y_train)
svm_reg_acc = svm_reg.score(X_test, y_test)
print("SVM Regression accuracy:", svm_reg_acc)

# Train Decision Tree
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
dt_clf_acc = dt_clf.score(X_test, y_test)
print("Decision Tree Classifier accuracy:", dt_clf_acc)

# Train Random Forest
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
rf_clf_acc = rf_clf.score(X_test, y_test)
print("Random Forest Classifier accuracy:", rf_clf_acc)

# Train Logistic Regression
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
lr_clf_acc = lr_clf.score(X_test, y_test)
print("Logistic Regression accuracy:", lr_clf_acc)

# Train KNN
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
knn_clf_acc = knn_clf.score(X_test, y_test)
print("KNN Classifier accuracy:", knn_clf_acc)

# Train Naive Bayes
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)
nb_clf_acc = nb_clf.score(X_test, y_test)
print("Naive Bayes Classifier accuracy:", nb_clf_acc)

svm_clf_y_pred = svm_clf.predict(X_test)
svm_reg_y_pred = svm_reg.predict(X_test)
dt_y_pred = dt.predict(X_test)
rf_y_pred = rf.predict(X_test)
lr_y_pred = lr.predict(X_test)
knn_y_pred = knn.predict(X_test)
nb_y_pred = nb.predict(X_test)


y_true = y_test

# calculate accuracy
svm_clf_accuracy = accuracy_score(y_true, svm_clf_y_pred)
svm_reg_accuracy = accuracy_score(y_true, svm_reg_y_pred)
dt_accuracy = accuracy_score(y_true, dt_y_pred)
rf_accuracy = accuracy_score(y_true, rf_y_pred)
lr_accuracy = accuracy_score(y_true, lr_y_pred)
knn_accuracy = accuracy_score(y_true, knn_y_pred)
nb_accuracy = accuracy_score(y_true, nb_y_pred)

# calculate precision 
svm_clf_precision = precision_score(y_true, svm_clf_y_pred, average='weighted')
svm_reg_precision = precision_score(y_true, svm_reg_y_pred, average='weighted')
dt_precision = precision_score(y_true, dt_y_pred, average='weighted')
rf_precision = precision_score(y_true, rf_y_pred, average='weighted')
lr_precision = precision_score(y_true, lr_y_pred, average='weighted')
knn_precision = precision_score(y_true, knn_y_pred, average='weighted')
nb_precision = precision_score(y_true, nb_y_pred, average='weighted')

# calculate recall
svm_clf_recall = recall_score(y_true, svm_clf_y_pred, average='weighted')
svm_reg_recall = recall_score(y_true, svm_reg_y_pred, average='weighted')
dt_recall = recall_score(y_true, dt_y_pred, average='weighted')
rf_recall = recall_score(y_true, rf_y_pred, average='weighted')
lr_recall = recall_score(y_true, lr_y_pred, average='weighted')
knn_recall = recall_score(y_true, knn_y_pred, average='weighted')
nb_recall = recall_score(y_true, nb_y_pred, average='weighted')

# calculate f1-score
svm_clf_f1 = f1_score(y_true, svm_clf_y_pred, average='weighted')
svm_reg_f1 = f1_score(y_true, svm_reg_y_pred, average='weighted')
dt_f1 = f1_score(y_true, dt_y_pred, average='weighted')
rf_f1 = f1_score(y_true, rf_y_pred, average='weighted')
lr_f1 = f1_score(y_true, lr_y_pred, average='weighted')
knn_f1 = f1_score(y_true, knn_y_pred, average='weighted')
nb_f1 = f1_score(y_true, nb_y_pred, average='weighted')



# Print svm classifier metrics
print("Accuracy: {:.2f}%".format(svm_clf_accuracy * 100))
print("Precision: {:.2f}%".format(svm_clf_precision * 100))
print("Recall: {:.2f}%".format(svm_clf_recall * 100))
print("F1-Score: {:.2f}%".format(svm_clf_f1 * 100))

# Print svm regression metrics
print("Accuracy: {:.2f}%".format(svm_reg_accuracy * 100))
print("Precision: {:.2f}%".format(svm_reg_precision * 100))
print("Recall: {:.2f}%".format(svm_reg_recall * 100))
print("F1-Score: {:.2f}%".format(svm_reg_f1 * 100))


# Print decision tree metrics
print("Accuracy: {:.2f}%".format(dt_accuracy * 100))
print("Precision: {:.2f}%".format(dt_precision * 100))
print("Recall: {:.2f}%".format(dt_recall * 100))
print("F1-Score: {:.2f}%".format(dt_f1 * 100))



# Print random forest metrics
print("Accuracy: {:.2f}%".format(rf_accuracy * 100))
print("Precision: {:.2f}%".format(rf_precision * 100))
print("Recall: {:.2f}%".format(rf_recall * 100))
print("F1-Score: {:.2f}%".format(rf_f1 * 100))


# Print logistics regression metrics
print("Accuracy: {:.2f}%".format(lr_accuracy * 100))
print("Precision: {:.2f}%".format(lr_precision * 100))
print("Recall: {:.2f}%".format(lr_recall * 100))
print("F1-Score: {:.2f}%".format(lr_f1 * 100))


# Print KNN metrics
print("Accuracy: {:.2f}%".format(knn_accuracy * 100))
print("Precision: {:.2f}%".format(lr_precision * 100))
print("Recall: {:.2f}%".format(knn_recall * 100))
print("F1-Score: {:.2f}%".format(knn_f1 * 100))


# Print Naive bayes
print("Accuracy: {:.2f}%".format(nb_accuracy * 100))
print("Precision: {:.2f}%".format(nb_precision * 100))
print("Recall: {:.2f}%".format(nb_recall * 100))
print("F1-Score: {:.2f}%".format(nb_f1 * 100))
