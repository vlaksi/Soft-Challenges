# import libraries here
import dlib
import cv2
from imutils import face_utils
from joblib import dump, load
from sklearn.svm import SVC
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))

def train_model(model, train_image_paths, train_image_labels):

    nbins = 9  # broj binova
    cell_size = (8, 8)  # broj piksela po celiji
    block_size = (3, 3)  # broj celija po bloku

    ulaz = []
    ulaz1 = []

    for img in train_image_paths:
        img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
        img = img[:500,:400]
        hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                          img.shape[0] // cell_size[0] * cell_size[0]),
                                _blockSize=(block_size[1] * cell_size[1],
                                            block_size[0] * cell_size[0]),
                                _blockStride=(cell_size[1], cell_size[0]),
                                _cellSize=(cell_size[1], cell_size[0]),
                                _nbins=nbins)
        ulaz.append(hog.compute(img))
        print(ulaz)
        ulaz1.append(img)
        print(ulaz1)
        break;


    ulaz = np.array(ulaz)
    #print(train_image_labels)
    izlaz = np.array(train_image_labels)
    ulaz = reshape_data(ulaz)
    print(ulaz)

    clf_svm = None
    #SVC(kernel='linear', probability=True)
    # clf_svm.fit(ulaz, izlaz)

    return clf_svm

def train_or_load_facial_expression_recognition_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """

    model = None

    print("Traniranje modela zapoceto.")
    model = train_model(model, train_image_paths, train_image_labels)
    print("Treniranje modela zavrseno.")

    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    return model


def extract_facial_expression_from_image(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje ekspresije lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati ekspresiju.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <String>  Naziv prediktovane klase (moguce vrednosti su: 'anger', 'contempt', 'disgust', 'happiness', 'neutral', 'sadness', 'surprise'
    """
    nbins = 9  # broj binova
    cell_size = (8, 8)  # broj piksela po celiji
    block_size = (3, 3)  # broj celija po bloku

    ulaz = []

    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    img = img[:500, :400]
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    ulaz.append(hog.compute(img))

    ulaz = np.array(ulaz)
    ulaz = reshape_data(ulaz)

    try:
        predict = trained_model.predict(ulaz)
        for rec in predict:
            facial_expression = rec
    except:
        facial_expression = "anger"
    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)

    return facial_expression
