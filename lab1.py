import sys
import numpy as np
import gzip
import os
from urllib.request import urlretrieve

TRAIN_IMAGES_NAME = "train-images-idx3-ubyte.gz"
TRAIN_LABELS_NAME = "train-labels-idx1-ubyte.gz"
TEST_IMAGES_NAME = "t10k-images-idx3-ubyte.gz"
TEST_LABELS_NAME = "t10k-labels-idx1-ubyte.gz"

SAMPLE_DATA_URL = "http://yann.lecun.com/exdb/mnist/"

SAMPLE_DATA_FOLDER = "./Data_folder/"

def download_archive(file_name, source, target_file):
    urlretrieve(source + file_name, target_file)

def load_mnist_images(images_file):
    with gzip.open(images_file, 'rb') as file:
        data = np.frombuffer(file.read(), np.uint8, offset=16)
        img_data = data.reshape(-1, 784)
    return img_data

def load_mnist_labels(labels_file):
    with gzip.open(labels_file, 'rb') as file:
        data = np.frombuffer(file.read(), np.uint8, offset=8)
    return data

def load_images_label_pair(images_file, labels_file):
    images_file_on_disk = SAMPLE_DATA_FOLDER + images_file
    if not os.path.exists(images_file_on_disk):
        download_archive(images_file, SAMPLE_DATA_URL, images_file_on_disk)
    images = load_mnist_images(images_file_on_disk)
    labels_file_on_disk = SAMPLE_DATA_FOLDER + labels_file
    if not os.path.exists(labels_file_on_disk):
        download_archive(labels_file, SAMPLE_DATA_URL, labels_file_on_disk)
    labels = load_mnist_labels(labels_file_on_disk)
    return images, labels

def retrieve_sample_data():
    if not os.path.exists(SAMPLE_DATA_FOLDER):
        os.makedirs(SAMPLE_DATA_FOLDER)
    train_imgs, train_lbls = load_images_label_pair(TRAIN_IMAGES_NAME, TRAIN_LABELS_NAME)
    test_imgs, test_lbls = load_images_label_pair(TEST_IMAGES_NAME, TEST_LABELS_NAME)
    return train_imgs, train_lbls, test_imgs, test_lbls

def normalize_mnist_image(image: np.array):
    return image / 255.0

def convert_number_to_array(y_list, classes):
    result = []
    for value in y_list:
        array_from_value = np.zeros(classes)
        array_from_value[value] = 1
        result.append(array_from_value)
    return np.array(result)

def sigmoid(x):
    return 1./(1. + np.exp(-x))

def sigmoid_grad(x):
    return x * (1. - x)

def softmax(x):
    max_value = np.max(x)
    shifted_args = x - max_value
    exps = np.exp(shifted_args)
    return exps / np.sum(exps)

def softmax_grad(x):
    return x * (1. - x)

def cross_entropy(predicted: np.array, expected: np.array):
    return -np.sum(expected.dot(np.log(predicted)) + (1. - expected).dot(np.log(1. - predicted)))


def cross_entropy_gradient(predicted: np.array, expected: np.array):
    return -expected / predicted - (1. - expected) / (1. - predicted)


def cross_entropy_error(predicted: np.array, expected: np.array):
    return predicted - expected

def find_precision(X_array: np.array, y_array: np.array, weightes_hidden_layer :np.array, weightes_output_layer :np.array):
    true_classifications = 0
    for x, y in zip(X_array, y_array):
        label = np.argmax(y)
        hidden_level = sigmoid(np.dot(weightes_hidden_layer, x))
        output_level = softmax(np.dot(weightes_output_layer, hidden_level))
        prediction = np.argmax(output_level)
        if (prediction == label):
            true_classifications += 1
    return true_classifications / X_array.shape[0]
 
def initialize_weights(size: np.array, val_range: np.array):
    return val_range[0] + (val_range[1]-val_range[0])*np.random.random((size[0], size[1]))

def main():

    np.random.seed(13)
    neuron_count = 200
    y_classes = 10
    max_epoch = 20
    max_precision = 0.98
    range_of_weights = (0.003, 0.007)
    learning_rate = 0.01

    print("Downloading sample data")
    train_data, train_labels, test_data, test_labels = retrieve_sample_data()
    print("Sample data normalization")
    X_train = np.array([normalize_mnist_image(np.array(image)) for image in train_data])
    y_train = convert_number_to_array(train_labels, y_classes)
    X_test = np.array([normalize_mnist_image(np.array(image)) for image in test_data])
    y_test = convert_number_to_array(test_labels, y_classes)
    print("Normalization done")

    x_params_count = X_train.shape[1]
    weightes_hidden_layer = initialize_weights((neuron_count, x_params_count), range_of_weights)    
    weightes_output_layer = initialize_weights((y_classes, neuron_count), range_of_weights)

    print("Training start")
    for epoch in range(max_epoch):
        for x, y in zip(X_train, y_train):

            weighted_input_hidden_layer = np.dot(weightes_hidden_layer, x)
            hidden_layer_output = sigmoid(weighted_input_hidden_layer)
            
            weighted_input_output_layer = np.dot(weightes_output_layer, hidden_layer_output)
            final_layer_output = softmax(weighted_input_output_layer)
            
            final_level_error = final_layer_output - y
            final_level_transferred_error = weightes_output_layer.T.dot(final_level_error)
            
            hidden_level_error = final_level_transferred_error * softmax_grad(hidden_layer_output)
            
            weightes_output_layer -= learning_rate * np.outer(final_level_error, hidden_layer_output)           
            weightes_hidden_layer -= learning_rate * np.outer(hidden_level_error, x)            
        
        current_precision = find_precision(X_test, y_test, weightes_hidden_layer, weightes_output_layer)
        print("  Step {0}: Precision {1}".format(epoch+1, current_precision))
        if (current_precision > max_precision):
            break
    print("Training finished")

    result_precision = find_precision(X_test, y_test, weightes_hidden_layer, weightes_output_layer)
    print("############################################")
    print("Total precision {0}".format(result_precision))
    print("############################################")

if __name__ == '__main__':
    sys.exit(main())