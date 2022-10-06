import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.filters import prewitt_v
from sklearn.svm import LinearSVC
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.utils import shuffle


def show_confusion_matrix(predictions, labels, title=None):
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.show()


# calculates the OOD performance
def ood_stats(predictions, labels, ood_label=90):
    ood_label = np.max(labels)
    acc = np.sum(np.logical_and(predictions == labels, labels == ood_label))
    print("OOD:", acc, "out of", 1000, ":", acc / 1000)
    return acc / 1000


def test_method(train_data, train_labels, test_data, test_labels, data_method=None):
    k = np.unique(test_labels).size
    if data_method:
        print("Testing", data_method)
    print("Randomforest:")
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    clf.fit(train_data, train_labels)
    print(clf.score(test_data, test_labels))
    ood_stats(clf.predict(test_data), test_labels)

    print("Adaboost")
    clf = AdaBoostClassifier(n_estimators=100, random_state=42)
    clf.fit(train_data, train_labels)
    print(clf.score(test_data, test_labels))
    ood_stats(clf.predict(test_data), test_labels)

    print("SVM")
    clf = make_pipeline(StandardScaler(),
                        LinearSVC(random_state=42, max_iter=150))
    clf.fit(train_data, train_labels)
    print(clf.score(test_data, test_labels))
    ood_stats(clf.predict(test_data), test_labels)

    print("MLP")
    clf = MLPClassifier(random_state=42, max_iter=150)
    clf.fit(train_data, train_labels)
    print(clf.score(test_data, test_labels))
    ood_stats(clf.predict(test_data), test_labels)

    print("KMeans")
    clf = KMeans(n_clusters=k)  # might require getting np.unique numbers if OOD
    clf.fit(train_data, train_labels)
    preds = clf.predict(test_data)
    print(accuracy_score(preds, test_labels))
    ood_stats(clf.predict(test_data), test_labels)


def get_data(OOD=False):
    cifar = tf.keras.datasets.cifar100
    (train_images, train_labels), (test_images, test_labels) = cifar.load_data()
    if OOD:
        return train_images, split_ood(train_labels.ravel()), test_images, split_ood(test_labels.ravel())
    return train_images, train_labels.ravel(), test_images, test_labels.ravel()


def split_ood(labels, ood_classes=None):
    if ood_classes is None:
        ood_classes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    ood_idx = labels == ood_classes[0]

    # get all the indexes that we will categorize as OOD
    for c in ood_classes:
        ood_idx = np.logical_or(ood_idx, labels == c)

    # fix the class numbers
    for i, n in enumerate(labels):
        reduction = np.sum(ood_classes < n)
        labels[i] -= reduction

    labels[ood_idx] = np.max(labels) + 1  # K + 1 class
    return labels


# A naive classification where we don't try to extract any features
def simple(train_images, train_labels, test_images, test_labels):
    # normalize images to 0-1
    train_images = train_images / 255
    test_images = test_images / 255

    # flatten the images
    train_images = train_images.reshape((train_images.shape[0], 32 * 32 * 3))
    test_images = test_images.reshape((test_images.shape[0], 32 * 32 * 3))

    test_method(train_images, train_labels, test_images, test_labels, "flatten/naive:")


# this will be a CNN for classification
def cnn_classification(train_images, train_labels, test_images, test_labels, epochs=15):
    k = np.unique(test_labels).size
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(k, activation='softmax'))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=epochs,
              validation_data=(test_images, test_labels))

    return model


# this will be a CNN to pass the data through in order to create new data! might require further testing
def cnn_dataline(train_images, train_labels, test_images, test_labels, model=None):
    if not model:
        model = cnn_classification(train_images, train_labels, test_images, test_labels, epochs=15)
    train_predictions = model.predict(train_images)
    print("Randomforest dataline:")
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    clf.fit(train_predictions, train_labels)
    print(clf.score(model.predict(test_data), test_labels))
    ood_stats(clf.predict(model.predict(test_data)), test_labels)

    return clf


def prewit(train_images, train_labels, test_images, test_labels):
    new_train = []
    for img in train_images:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        output = prewitt_v(gray_img)

        new_train.append(output)

    train_images = np.array(new_train).reshape((50000, 32 * 32))

    new_test = []
    for img in test_images:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        output = prewitt_v(gray_img)
        new_test.append(output)

    test_images = np.array(new_test).reshape((10000, 32 * 32))
    test_method(train_images, train_labels, test_images, test_labels)



def history_of_gradients(train_images, train_labels, test_images, test_labels):
    from skimage.feature import hog
    new_train = []
    for img in train_images:
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), feature_vector=True, block_norm='L2-Hys', transform_sqrt=True,
                            visualize=True, channel_axis=-1)
        new_train.append(fd)

    train_images = np.array(new_train)
    train_images, train_labels = shuffle(train_images, train_labels, random_state=42)

    new_test = []
    for img in test_images:
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), feature_vector=True, block_norm='L2-Hys', transform_sqrt=True,
                            visualize=True, channel_axis=-1)
        new_test.append(fd)

    # technically not images
    test_images = np.array(new_test)
    test_images, test_labels = shuffle(test_images, test_labels, random_state=42)
    test_method(train_images, train_labels, test_images, test_labels, "Histogram of Gradients")






print("With OOD:")
train_data, train_labels, test_data, test_labels = get_data(OOD=True)
#
history_of_gradients(train_data, train_labels, test_data, test_labels)
simple(train_data, train_labels, test_data, test_labels)
prewit(train_data, train_labels, test_data, test_labels)

m = cnn_classification(train_data, train_labels, test_data, test_labels)
print(m.evaluate(test_data, test_labels))
preds = np.argmax(m.predict(test_data), axis=-1)
ood_stats(preds, test_labels)
cnn_dataline(train_data, train_labels, test_data, test_labels, m)


print("Without OOD:")
train_data, train_labels, test_data, test_labels = get_data(OOD=False)

history_of_gradients(train_data, train_labels, test_data, test_labels)
prewit(train_data, train_labels, test_data, test_labels)
simple(train_data, train_labels, test_data, test_labels)
m = cnn_classification(train_data, train_labels, test_data, test_labels)
print("cnn: ", m.evaluate(test_data, test_labels))
cnn_dataline(train_data, train_labels, test_data, test_labels, m)
