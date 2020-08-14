import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict


def generate_features_targets(data):
    """
    Splits the data into the features and targets.
    :param data: ndarray
    :return: input_features ndarray, output_targets ndarray
    """
    output_targets = np.empty(shape=len(data), dtype='<U20')
    output_targets[:] = data['class']

    input_features = np.empty(shape=(len(data), 13))
    input_features[:, 0] = data['u-g']
    input_features[:, 1] = data['g-r']
    input_features[:, 2] = data['r-i']
    input_features[:, 3] = data['i-z']
    input_features[:, 4] = data['ecc']
    input_features[:, 5] = data['m4_u']
    input_features[:, 6] = data['m4_g']
    input_features[:, 7] = data['m4_r']
    input_features[:, 8] = data['m4_i']
    input_features[:, 9] = data['m4_z']
    input_features[:, 10] = data['petroR50_u'] / data['petroR90_u']
    input_features[:, 11] = data['petroR50_r'] / data['petroR90_r']
    input_features[:, 12] = data['petroR50_z'] / data['petroR90_z']

    return input_features, output_targets


def splitdata_train_test(data, training_fraction):
    """
    Splits the data into training and testing sets according to the fraction.
    :param data: array of galaxy data
    :param training_fraction: float between 0 and 1 with the fraction of data used for training
    :return: training data array, testing data array
    """
    np.random.seed(8)
    np.random.shuffle(data)
    split = int(len(data) * training_fraction)
    training = data[:split]
    testing = data[split:]
    return training, testing


def dtc_predict_actual(data, fraction):
    """
    Predicts the class of the galaxies
    :param data: array of galaxy data
    :param fraction: fraction of data used for training, float between 0 and 1
    :return: predictions array of classes, test_targets array giving true classes
    """
    training, testing = splitdata_train_test(data, fraction)
    train_features, train_targets = generate_features_targets(training)
    test_features, test_targets = generate_features_targets(testing)
    dtc = DecisionTreeClassifier()
    dtc.fit(train_features, train_targets)
    predictions = dtc.predict(test_features)
    return predictions, test_targets


def calculate_accuracy(predicted, actual):
    correct = 0
    for i, j in zip(predicted, actual):
        if i == j:
            correct += 1
    return correct / len(predicted)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plots a confusion matrix
    :param cm: confusion matrix
    :param classes: a list of the classes of galaxy
    :param normalize: boolean for whether to normalize the values
    :param title: the title of the confusion matrix plot
    :param cmap: the colour map used for the confusion matrix plot
    :return: None
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Non-normalized confusion matrix')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{}".format(cm[i, j]), horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')


def rf_predict_actual(data, n_estimators):
    """
    A prediction of the galaxy class using a random forest
    :param data: array of galaxies
    :param n_estimators: number of trees to use
    :return: predicted class array, targets class array
    """
    features, targets = generate_features_targets(data)
    rfc = RandomForestClassifier(n_estimators=n_estimators)
    predicted = cross_val_predict(rfc, features, targets, cv=10)
    return predicted, targets
