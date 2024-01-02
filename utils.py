import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from collections import Counter
import seaborn as sns

import visualkeras
from collections import defaultdict
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, UpSampling2D, InputLayer, Reshape
from PIL import ImageFont

def import_images(labels: list[str], 
                  img_size: int = 128, 
                  validate_size:float = 0.2, 
                  flat: bool = False, 
                  categorical: bool = True, 
                  mono: bool = False,
                  shuffle_test: bool = True) -> tuple:
    """import images from achive

    Args:
        labels (list[str]): labels name liset
        img_size (int, optional): length of square image. Defaults to 128.
        validate_size (float, optional): validate ratio. Defaults to 0.2.
        flat (bool, optional): whether to flatten the image. Defaults to False.
        categorical (bool, optional): whether to one-hot encoding. Defaults to True.

    Returns:
        tuple: x_train, x_validate, x_test, y_train, y_validate, y_test
    """
    # Train data
    x_train = []
    y_train = []
    for label in labels:
        data_path = os.path.join(".", "archive", "Training", label)
        for img_file in tqdm(os.listdir(data_path)):
            if mono:
                img = cv2.imread(os.path.join(data_path, img_file), 0)
            else:
                img = cv2.imread(os.path.join(data_path, img_file))
            img = cv2.resize(img, (img_size, img_size))
            x_train.append(img)
            y_train.append(label)
    # Train label
    if categorical:
        for index in range(len(y_train)):
            y_train[index] = labels.index(y_train[index])
        y_train = to_categorical(y_train, len(labels))
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    if flat:
        x_train = x_train.reshape((x_train.shape[0], -1))
    x_train, y_train = shuffle(x_train, y_train, random_state=4)

    # Validate data split
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=validate_size)

    # Test data
    x_test = []
    y_test = []
    for label in labels:
        data_path = os.path.join(".", "archive", "Testing", label)
        for img_file in tqdm(os.listdir(data_path)):
            if mono:
                img = cv2.imread(os.path.join(data_path, img_file), 0)
            else:
                img = cv2.imread(os.path.join(data_path, img_file))
            img = cv2.resize(img, (img_size, img_size))
            x_test.append(img)
            y_test.append(label)
    # Test label
    if categorical:
        for index in range(len(y_test)):
            y_test[index] = labels.index(y_test[index])
        y_test = to_categorical(y_test, len(labels))

    x_test, y_test = np.array(x_test), np.array(y_test)
    if flat:
        x_test = x_test.reshape((x_test.shape[0], -1))
    if shuffle_test:
        x_test, y_test = shuffle(x_test, y_test, random_state=4)
    return (x_train, x_validate, x_test, y_train, y_validate, y_test)

def plot_images(x: np.ndarray, y=[], label=False, nrow=5, ncol=5, randomize=True):
    """
    Plot Images
    """
    fig, ax = plt.subplots(nrow, ncol, figsize=(5*ncol, 4*nrow))
    if label:
        x = x[y==label]
        y = y[y==label]
    num_samples = x.shape[0]

    for k, a in enumerate(ax.ravel()):
        if randomize:
            j = np.random.choice(num_samples)
        else:
            j = k
        if x.ndim == 4:
            a.imshow(x[j])
        elif x.ndim == 3:
            a.imshow(x[j], cmap='Greys_r')
        elif x.ndim == 2:
            img_size = int(np.sqrt(x.shape[1]))
            assert img_size * img_size == x.shape[1]
            a.imshow(x[j].reshape((img_size, img_size)), cmap='Greys_r')
        a.set_xticks([])
        a.set_yticks([])
        # check if there is label
        if len(y):
            a.set_title(f"Label: {y[j]}")
            
def plot_labels(labels, str='Train'):
    colors = sns.color_palette("Blues")
    fig, ax = plt.subplots()
    labels_count = Counter(labels)
    ax.pie(x=list(labels_count.values()),
           labels=list(labels_count.keys()),
           startangle=90, 
           explode=[0,0,0.1,0],
           autopct="%1.1f%%",
           colors=colors)
    ax.axis('equal')
    ax.set_title(f'Rate of Tumor Types ({str})')
    plt.show()

def plot_nn_history(history: dict, y_max_lim = 1.0):
    acc = [0.] + history['accuracy']
    val_acc = [0.] + history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, y_max_lim])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
    
def plot_nn_structure(model, max_xy=300, max_z=200, scale_xy=0.8):
    color_map = defaultdict(dict)
    color_map[InputLayer]['fill'] = '#2683c6'#'#1cade4'#'#2683c6','#28c4cc'
    color_map[Conv2D]['fill'] = '#7ec492'
    color_map[Dropout]['fill'] = 'grey'
    color_map[MaxPooling2D]['fill'] = '#1cade4'
    color_map[UpSampling2D]['fill'] = '#1cade4'
    color_map[Dense]['fill'] = 'green'
    color_map[Flatten]['fill'] = '#bcdbf2'
    color_map[Reshape]['fill'] = 'grey'
    font = ImageFont.truetype("arial.ttf", 13)

    img = visualkeras.layered_view(model, 
                             legend=True, 
                             max_xy=max_xy, 
                             max_z=max_z,
                             scale_xy=scale_xy, 
                             font=font, 
                             color_map=color_map)
    return img