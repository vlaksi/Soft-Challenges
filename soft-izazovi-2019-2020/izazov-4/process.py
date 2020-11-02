# import libraries here
import pickle

import cv2
import dlib
import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC, LinearSVC
from imblearn.over_sampling import RandomOverSampler


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))

def train_or_load_traffic_sign_model(train_positive_images_paths, train_negative_images_path, train_image_labels):
    """
    Procedura prima listu putanja do pozitivnih i negativnih fotografija za obucavanje, liste
    labela za svaku fotografiju iz pozitivne liste

    Procedura treba da istrenira model(e) i da ih sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model(e) ako on nisu istranirani, ili da ih samo ucita ako su prethodno
    istrenirani i ako se nalaze u folderu za serijalizaciju

    :param train_positive_images_paths: putanje do pozitivnih fotografija za obucavanje
    :param train_negative_images_path: putanje do negativnih fotografija za obucavanje
    :param train_image_labels: labele za pozitivne fotografije iz liste putanja za obucavanje - tip znaka
    :return: lista modela
    """
    # TODO - Istrenirati modele ako vec nisu istrenirani, ili ih samo ucitati iz foldera za serijalizaciju
    pos_features = []
    neg_features = []
    labels = []
    labels1 = []
    labels2 = []
    models = []

    nbins = 9  # broj binova
    cell_size = (8, 8)  # broj piksela po celiji
    block_size = (3, 3)  # broj celija po bloku
    #print(train_positive_images_paths)


    for img_path in train_positive_images_paths:
        #print(img_path)
        img = load_image(img_path)
        img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
        hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                          img.shape[0] // cell_size[0] * cell_size[0]),
                                _blockSize=(block_size[1] * cell_size[1],
                                            block_size[0] * cell_size[0]),
                                _blockStride=(cell_size[1], cell_size[0]),
                                _cellSize=(cell_size[1], cell_size[0]),
                                _nbins=nbins)
        pos_features.append(hog.compute(img))
        # cv2.imshow('bla1', img)
        # cv2.waitKey(0)

    for label in train_image_labels:
        labels.append(label[0])

    pos_features = np.array(pos_features)
    ulaz_prvi_model = reshape_data(pos_features)
    izlaz_prvi_model = np.array(labels)
    filename1 = '.\\serialization_folder\\finalized_model1.sav'
    try:
        models0 = joblib.load(filename1)
        print("usao_prvi_model")
        #y_test_pred = models[0].predict(ulaz_prvi_model)
        #print(y_test_pred)
    except:
        models0 = SVC(kernel='linear', probability=True)
        models0.fit(ulaz_prvi_model, izlaz_prvi_model)
        joblib.dump(models0, filename1)
        print("usao1_prvi_model")
        #y_test_pred = models0.predict(ulaz_prvi_model)
        #print(y_test_pred)


    for img_path in train_negative_images_path:
        #print(img_path)
        img = load_image(img_path)
        img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
        hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                          img.shape[0] // cell_size[0] * cell_size[0]),
                                _blockSize=(block_size[1] * cell_size[1],
                                            block_size[0] * cell_size[0]),
                                _blockStride=(cell_size[1], cell_size[0]),
                                _cellSize=(cell_size[1], cell_size[0]),
                                _nbins=nbins)
        neg_features.append(hog.compute(img))
        labels.append('NEGATIVNO')
        # cv2.imshow('bla', img)
        # cv2.waitKey(0)

    neg_features = np.array(neg_features)
    ulaz_drugi_model = np.vstack((pos_features, neg_features))
    #print(x)

    for label in labels:
        if label != 'NEGATIVNO':
            labels1.append('jeste_znak')
            labels2.append(0)
        else:
            labels1.append('nije_znak')
            labels2.append(1)

    izlaz_drugi_model = np.array(labels1)

    ulaz_drugi_model = reshape_data(ulaz_drugi_model)
    #print(x)
    filename2 = '.\\serialization_folder\\finalized_model2.sav'

    try:
        models1 = joblib.load(filename2)
        print("usao_drugi_model")
        #y_test_pred = models1.predict(x)
        #print(y_test_pred)
    except:
        models1 = SVC(kernel='linear', probability=True)

        ros = RandomOverSampler(random_state=42)
        print(ulaz_drugi_model.shape)
        ulaz_drugi_model, izlaz_drugi_model = ros.fit_resample(ulaz_drugi_model, izlaz_drugi_model)
        print(ulaz_drugi_model.shape)

        models1.fit(ulaz_drugi_model, izlaz_drugi_model)
        joblib.dump(models1, filename2)
        print("usao1_drugi_model")
        #y_test_pred = models1.predict(x)
        #print(y_test_pred)


    models.append(models1) #jel jeste znak il nije
    models.append(models0) #tacno koji je znak

    filename3 = '.\\serialization_folder\\model3.pickle'
    izlaz_treci_model = np.array(labels2)
    ulaz_treci_model = np.vstack((pos_features, neg_features))
    ulaz_treci_model = reshape_data(ulaz_treci_model)
    ros = RandomOverSampler(random_state=42)
    ulaz_treci_model, izlaz_treci_model = ros.fit_resample(ulaz_treci_model, izlaz_treci_model)

    try:
        #models2 = joblib.load(filename3)
        supportVectors = pickle.load(open(filename3, 'rb'))
    except:
        #models2 = LinearSVC()
        #models2.fit(ulaz_treci_model, izlaz_treci_model)
        models2 = cv2.ml.SVM_create()
        models2.setType(cv2.ml.SVM_C_SVC)
        models2.setKernel(cv2.ml.SVM_LINEAR)
        models2.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        models2.train(ulaz_treci_model, cv2.ml.ROW_SAMPLE, izlaz_treci_model)
        print('usao_treci_model')

        (rho, alpha, supportVectorIndices) = models2.getDecisionFunction(0)
        supportVectors = models2.getSupportVectors()
        supportVectors = np.append(supportVectors, -rho)
        #joblib.dump(models2, filename3)
        pickle.dump(supportVectors, open(filename3, 'wb'))

    models.append(supportVectors)

    return models


def detect_traffic_signs_from_image(trained_models, image_path):
    """
    Procedura prima listu istreniranih modela za detekciju i klasifikaciju saobracajnih znakova i putanju do fotografije na kojoj
    se nalazi novi znakovi koje treda detektovati i klasifikovati

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_models: Istreniranih modela za detekciju i klasifikaciju saobracajnih znakova
    :param image_path: Putanja do fotografije sa koje treba detektovati 
    :return: Naziv prediktovanog tipa znaka, koordinate detektovanog znaka
    """
    # print(image_path)
    # (rects, weights) = cv2.CascadeClassifier.detectMultiScale(cv2.imread(image_path), 2)
    # for (x, y, w, h) in rects:
    #     cv2.rectangle(cv2.imread(image_path), (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the output image
    # cv2.imshow("Detections", cv2.imread(image_path))
    # cv2.waitKey(0)
    # TODO - Detektovati saobracajne znakove i vratiti listu detektovanih znakova:
    # za 2 znaka primer povratne vrednosti[[10, 15, 20, 20, "ZABRANA"], [30, 40, 60, 70, "DRUGI"]]

    nbins = 9  # broj binova
    cell_size = (8, 8)  # broj piksela po celiji
    block_size = (3, 3)  # broj celija po bloku


    img = cv2.imread(image_path, 0)
    # img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
    hog = cv2.HOGDescriptor(_winSize=(60 // cell_size[1] * cell_size[1],
                                      60 // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    # detector = trained_models[0].coef_
    hog.setSVMDetector(trained_models[2])

    (rects, weights) = hog.detectMultiScale(img, 0, winStride=(10, 10),
                                            padding=(2, 2), scale=1.001, useMeanshiftGrouping=True)
    #print(rects)
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the output image
    cv2.imshow("Detections", img)
    cv2.waitKey(0)

    niz = []
    detections = []

    image = load_image(image_path)
    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        image1 = image[y:y+h, x:x+w]
        image1 = cv2.resize(image1, (60, 60), interpolation=cv2.INTER_AREA)
        hog = cv2.HOGDescriptor(_winSize=(image1.shape[1] // cell_size[1] * cell_size[1],
                                          image1.shape[0] // cell_size[0] * cell_size[0]),
                                _blockSize=(block_size[1] * cell_size[1],
                                            block_size[0] * cell_size[0]),
                                _blockStride=(cell_size[1], cell_size[0]),
                                _cellSize=(cell_size[1], cell_size[0]),
                                _nbins=nbins)
        ulaz_prvi_model = []
        ulaz_prvi_model.append(hog.compute(image1))
        ulaz_prvi_model = np.array(ulaz_prvi_model)

        ulaz_prvi_model = reshape_data(ulaz_prvi_model)

        izlaz_prvi_model = trained_models[0].predict(ulaz_prvi_model)

        if izlaz_prvi_model[0] == 'jeste_znak':
            print('usao_jeste_znak')
            izlaz_drugi_model = trained_models[1].predict(ulaz_prvi_model)
            niz.append(x)
            niz.append(y)
            niz.append(x + w)
            niz.append(y + h)
            niz.append(izlaz_drugi_model[0])  # proveriti dal trebaju kockaste
            detections.append(niz)
            niz = []

    print(detections)
    #detections = [[0, 0, 0, 0, "DRUGI"]]  # x_min, y_min, x_max, y_max, tip znaka
    return detections
