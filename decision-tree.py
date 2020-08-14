from support_funcs import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


data = np.load('galaxy_catalogue.npy')

predicted, targets = dtc_predict_actual(data, 0.7)

model_score = calculate_accuracy(predicted, targets)
print("Accuracy: " + str(model_score))

class_labels = list(set(targets))
model_cm = confusion_matrix(y_true=targets, y_pred=predicted, labels=class_labels)

plt.figure()
plot_confusion_matrix(model_cm, classes=class_labels, normalize=False, title="Decision Tree Confusion Matrix")
plt.savefig('figures/decision_tree_matrix.png', dpi=200, pad_inches=0.3)
plt.show()
