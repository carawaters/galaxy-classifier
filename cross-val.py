import numpy as np
from support_funcs import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

data = np.load('galaxy_catalogue.npy')

features, targets = generate_features_targets(data)

# Using cross validation of the decision tree classifier rather than splitting by a given fraction
dtc = DecisionTreeClassifier()
predicted = cross_val_predict(dtc, features, targets, cv=10)

model_score = calculate_accuracy(predicted, targets)
print("Accuracy: " + str(model_score))

class_labels = list(set(targets))
model_cm = confusion_matrix(y_true=targets, y_pred=predicted, labels=class_labels)

plt.figure()
plot_confusion_matrix(model_cm, classes=class_labels, normalize=False, title="Cross Validation Confusion Matrix")
plt.savefig('figures/cross_val_matrix.png', dpi=200, pad_inches=0.3)
plt.show()
