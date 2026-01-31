import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'smarket.csv')

df = pd.read_csv(file_path)

train_data = df[df['Year'] < 2005]
test_data = df[df['Year'] == 2005]
X_train = train_data[['Lag1', 'Lag2']]
y_train = train_data['Direction']
X_test = test_data[['Lag1', 'Lag2']]
y_test = test_data['Direction']

print("LDA Results")
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)

print("Priors:", lda_model.priors_)
print("Group Means:\n", lda_model.means_)
print("Coefficients:\n", lda_model.coef_)

lda_pred = lda_model.predict(X_test)
lda_cm = confusion_matrix(y_test, lda_pred)
lda_acc = accuracy_score(y_test, lda_pred)

print("Confusion Matrix:\n", lda_cm)
print("Accuracy:", lda_acc)

print("\nQDA Results")
qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(X_train, y_train)

print("Priors:", qda_model.priors_)
print("Group Means:\n", qda_model.means_)

qda_pred = qda_model.predict(X_test)
qda_cm = confusion_matrix(y_test, qda_pred)
qda_acc = accuracy_score(y_test, qda_pred)

print("Confusion Matrix:\n", qda_cm)
print("Accuracy:", qda_acc)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.matshow(lda_cm, cmap='Blues', alpha=0.5)
ax1.set_title('LDA Confusion Matrix')
ax1.set_ylabel('True Label')
ax1.set_xlabel('Predicted Label')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Down', 'Up'])
ax1.set_yticklabels(['Down', 'Up'])

for i in range(2):
    for j in range(2):
        ax1.text(j, i, str(lda_cm[i, j]), ha='center', va='center')

ax2.matshow(qda_cm, cmap='Greens', alpha=0.5)
ax2.set_title('QDA Confusion Matrix')
ax2.set_ylabel('True Label')
ax2.set_xlabel('Predicted Label')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Down', 'Up'])
ax2.set_yticklabels(['Down', 'Up'])

for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(qda_cm[i, j]), ha='center', va='center')

plt.tight_layout()
plt.show()