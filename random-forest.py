import numpy as np
import matplotlib.pyplot as plt
from support_funcs import *
from sklearn.metrics import confusion_matrix

data = np.load('galaxy_catalogue.npy')

number_estimators = 50  # no trees
predicted, actual = rf_predict_actual(data, number_estimators)

accuracy = calculate_accuracy(predicted, actual)
print("Accuracy score:", accuracy)

class_labels = list(set(actual))
model_cm = confusion_matrix(y_true=actual, y_pred=predicted, labels=class_labels)

plt.figure()
plot_confusion_matrix(model_cm, classes=class_labels, normalize=False, title="Random Forest Confusion Matrix")
plt.savefig('figures/random_forest_matrix.png', dpi=500, pad_inches=0.3)
plt.show()
