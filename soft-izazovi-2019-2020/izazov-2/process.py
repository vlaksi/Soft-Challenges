# import libraries here

import cv2
import numpy as np
from fuzzywuzzy import fuzz
from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.cluster import KMeans

def select_roi_za_manja_slova(image_orig, image_bin):
    '''
    Funkcija kao u vežbi 2, iscrtava pravougaonike na originalnoj slici, pronalazi sortiran niz regiona sa slike,
    i dodatno treba da sačuva rastojanja između susednih regiona.
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Način određivanja kontura je promenjen na spoljašnje konture: cv2.RETR_EXTERNAL
    regions_array = []
    regions_array1 = []
    regions_array2 = []
    indexi = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        region = image_bin[y:y+h+1,x:x+w+1]
        if h < 30 or w < 30:
            """region = image_bin[y:y + h + 100, x-14:x + w + 14]
            try:
                regions_array.append([resize_region(region), (x-14, y, w + 14, h + 60)])
            except:
                regions_array.append([region, (x - 14, y, w + 14, h + 60)])"""
            continue
        else:
            try:
                regions_array.append([resize_region(region), (x, y, w, h)])
            except:
                regions_array.append([region, (x, y, w, h)])
        #cv2.imshow('jjj', region)
        #cv2.imshow('jdcdfdfj', resize_region(region))
        #cv2.waitKey(0)

        cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)

    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    for i in range(0,len(regions_array)):
        for j in range(0,len(regions_array)):
            if i == j:
                continue
            temp1 = regions_array[i]
            temp2 = regions_array[j]
            for pixel1 in range(temp1[1][0], temp1[1][0] + temp2[1][2]):
                for pixel2 in range(temp2[1][0], temp2[1][0] + temp2[1][2]):
                    if pixel1 == pixel2:
                        if temp2[1][3] > temp1[1][3]:
                             regions_array1.append(temp1)
                        else:
                             regions_array1.append(temp2)
                break


    for i in range(0,len(regions_array)):  #trazim index duplikata
        for j in range(0,len(regions_array1)):
            if i == j:
                continue
            temp1 = regions_array[i]
            temp2 = regions_array1[j]
            if temp1[1] == temp2[1]:
                indexi.append(i)
                #print(i)

    for ele in sorted(indexi, reverse=True): #brisem duplikate C, S..
        try:
            del regions_array[ele]
        except:
            print("ne brisu se duplikati")

    print(len(regions_array))
    #print(regions_array)
    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    print(sorted_rectangles)
    region_distances = []
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles)-1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index+1]
        distance = next_rect[0] - (current[0]+current[2]) #X_next - (X_current + W_current)
        region_distances.append(distance)

    return sorted_rectangles, image_orig, sorted_regions, region_distances

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]
def resize_region(region):
    resized = cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
    return resized
def scale_to_range(image):
    return image / 255
def matrix_to_vector(image):
    return image.flatten()
def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann
def convert_output(outputs):
    return np.eye(len(outputs))

def select_roi(image_orig, image_bin):
    '''
    Funkcija kao u vežbi 2, iscrtava pravougaonike na originalnoj slici, pronalazi sortiran niz regiona sa slike,
    i dodatno treba da sačuva rastojanja između susednih regiona.
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Način određivanja kontura je promenjen na spoljašnje konture: cv2.RETR_EXTERNAL
    regions_array = []
    regions_array1 = []
    regions_array2 = []
    indexi = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        region = image_bin[y:y+h+1,x:x+w+1]
        if h < 30 or w < 30:
            continue
        elif h < 80 and w < 90:
            region = image_bin[y:y + h + 190, x-20:x + w + 25]
            try:
                regions_array.append([resize_region(region), (x-20, y, w + 25, h + 190)])
            except:
                regions_array.append([region, (x - 20, y, w + 25, h + 190)])
        else:
            try:
                regions_array.append([resize_region(region), (x, y, w, h)])
            except:
                regions_array.append([region, (x, y, w, h)])
        #cv2.imshow('jjj', region)
        #cv2.imshow('jdcdfdfj', resize_region(region))
        #cv2.waitKey(0)

        cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)

    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    for i in range(0,len(regions_array)):
        for j in range(0,len(regions_array)):
            if i == j:
                continue
            temp1 = regions_array[i]
            temp2 = regions_array[j]
            for pixel1 in range(temp1[1][0], temp1[1][0] + temp2[1][2]):
                for pixel2 in range(temp2[1][0], temp2[1][0] + temp2[1][2]):
                    if pixel1 == pixel2:
                        if temp2[1][3] > temp1[1][3]:
                             regions_array1.append(temp1)
                        else:
                             regions_array1.append(temp2)
                break


    for i in range(0,len(regions_array)):  #trazim index duplikata
        for j in range(0,len(regions_array1)):
            if i == j:
                continue
            temp1 = regions_array[i]
            temp2 = regions_array1[j]
            if temp1[1] == temp2[1]:
                indexi.append(i)
                #print(i)

    for ele in sorted(indexi, reverse=True): #brisem duplikate C, S..
        try:
            del regions_array[ele]
        except:
            print("ne brisu se duplikati")

    print(len(regions_array))
    #print(regions_array)
    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    print(sorted_rectangles)
    region_distances = []
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles)-1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index+1]
        distance = next_rect[0] - (current[0]+current[2]) #X_next - (X_current + W_current)
        region_distances.append(distance)

    return sorted_rectangles, image_orig, sorted_regions, region_distances

def create_ann():
    '''
    Implementirati veštačku neuronsku mrežu sa 28x28 ulaznih neurona i jednim skrivenim slojem od 128 neurona.
    Odrediti broj izlaznih neurona. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    # Postaviti slojeve neurona mreže 'ann'
    ann.add(Dense(500, input_dim=784, activation='sigmoid'))
    ann.add(Dense(128, input_dim=500, activation='sigmoid'))
    ann.add(Dense(60, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.5, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=500, batch_size=1, verbose=0, shuffle=False)

    return ann


def serialize_ann(ann, serialization_folder):
    # serijalizuj arhitekturu neuronske mreze u JSON fajl
    model_json = ann.to_json()
    with open(serialization_folder + "neuronska.json", "w") as json_file:
        json_file.write(model_json)
    # serijalizuj tezine u HDF5 fajl
    ann.save_weights(serialization_folder + "neuronska.h5")


def load_trained_ann(serialization_folder):
    try:
        # Ucitaj JSON i kreiraj arhitekturu neuronske mreze na osnovu njega
        json_file = open(serialization_folder + "neuronska.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ann = model_from_json(loaded_model_json)
        # ucitaj tezine u prethodno kreirani model
        ann.load_weights(serialization_folder + "neuronska.h5")
        print("Istrenirani model uspesno ucitan.")
        return ann
    except Exception as e:
        # ako ucitavanje nije uspelo, verovatno model prethodno nije serijalizovan pa nema odakle da bude ucitan
        return None

def image_gray(image):
    #cv2.imshow('grey', cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    #cv2.waitKey(0)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_binT(image_gs):
    ret, image_bin = cv2.threshold(image_gs, 220, 255, cv2.THRESH_BINARY)
    return image_bin
def image_binE(image_gs):
    ret, image_bin = cv2.threshold(image_gs, 215, 255, cv2.THRESH_OTSU)
    return image_bin
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def display_result(outputs, alphabet, k_means):
    w_space_group = max(enumerate(k_means.cluster_centers_), key = lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:,:]):
        if (k_means.labels_[idx] == w_space_group):
            result += ' '
        result += alphabet[winner(output)]
    return result

def train_or_load_character_recognition_model(train_image_paths, serialization_folder):
    """
    Procedura prima putanje do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija alfabeta), kao i
    putanju do foldera u koji treba sacuvati model nakon sto se istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija alfabeta
    :param serialization_folder: folder u koji treba sacuvati serijalizovani model
    :return: Objekat modela
    """
    letters = []
    for imagee in train_image_paths:
        image_color = load_image(imagee)
        img = image_binT(image_gray(image_color))
        img = 255 - img
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((11, 11), np.uint8)
        dilation = cv2.dilate(opening, kernel, iterations=1)
        blurred = cv2.GaussianBlur(src=dilation,ksize=(5, 5), sigmaX=0)
        #cv2.imshow('img', cv2.resize(blurred, (1860, 110)))
        sorted_rectangles, selected_regions, letters1, region_distances = select_roi(image_color.copy(), blurred)
        letters = letters + letters1
        #print(letters)
        #cv2.imshow('bla', selected_regions)
        #cv2.waitKey(0)

    print(len(letters))

    alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'Š', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z', 'Ž', 'a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']
    inputs = prepare_for_ann(letters)
    outputs = convert_output(alphabet)
    ann = load_trained_ann(serialization_folder)

    if ann == None:
        print("Traniranje modela zapoceto.")
        ann = create_ann()
        ann = train_ann(ann, inputs, outputs)
        print("Treniranje modela zavrseno.")
        serialize_ann(ann, serialization_folder)


    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    #model = None
    return ann

def getList(dict):
    lista = []
    for key in dict.keys():
        lista.append(key)

    return lista

def extract_text_from_image(trained_model, image_path, vocabulary):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje znakova (karaktera), putanju do fotografije na kojoj
    se nalazi tekst za ekstrakciju i recnik svih poznatih reci koje se mogu naci na fotografiji.
    Procedura treba da ucita fotografiju sa prosledjene putanje, i da sa nje izvuce sav tekst koriscenjem
    openCV (detekcija karaktera) i prethodno istreniranog modela (prepoznavanje karaktera), i da vrati procitani tekst
    kao string.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba procitati tekst.
    :param vocabulary: <Dict> Recnik SVIH poznatih reci i ucestalost njihovog pojavljivanja u tekstu
    :return: <String>  Tekst procitan sa ulazne slike
    """

    img_hsv = cv2.cvtColor(cv2.imread(image_path).copy(), cv2.COLOR_BGR2HSV)
    lower_red = np.array([20, 100, 170])
    upper_red = np.array([80, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    #cv2.imshow('HSV0', mask0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    img_dil = cv2.dilate(mask0, kernel, iterations=1)
    img_close = cv2.erode(img_dil, kernel, iterations=2)
    img_dil = cv2.dilate(img_close, kernel, iterations=1)
    blurred = cv2.GaussianBlur(src=img_dil, ksize=(5, 5), sigmaX=0)
    #cv2.imshow('2222222222', blurred)

    coords = np.column_stack(np.where(blurred > 0))
    angle = cv2.minAreaRect(coords)[-1]
    print(angle)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    # rotate the image to deskew it
    (h, w) = cv2.imread(image_path).shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(cv2.imread(image_path), M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # show the output image
    #print("[INFO] angle: {:.3f}".format(angle))
    #cv2.imshow("Input", cv2.imread(image_path))
    #cv2.imshow("Rotated", rotated)
    #cv2.waitKey(1)

    img_hsv = cv2.cvtColor(rotated.copy(), cv2.COLOR_BGR2HSV)
    lower_red = np.array([20, 120, 170])
    upper_red = np.array([80, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    #cv2.imshow('mask0', mask0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    opening = cv2.morphologyEx(mask0, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((10, 10), np.uint8)
    dilation = cv2.dilate(opening, kernel, iterations=1)
    blurred = cv2.GaussianBlur(src=dilation, ksize=(5, 5), sigmaX=0)
    #cv2.imshow('HSV1', blurred)
    #cv2.waitKey(1)

    sorted_rectangles, selected_regions, letters, distances = select_roi(cv2.imread(image_path).copy(), blurred)
    if len(sorted_rectangles) < 25:
        img_hsv = cv2.cvtColor(rotated.copy(), cv2.COLOR_BGR2HSV)
        lower_red = np.array([20, 120, 180])
        upper_red = np.array([80, 255, 255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
        #cv2.imshow('HSV0', mask0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        opening = cv2.morphologyEx(mask0, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(opening, kernel, iterations=1)
        blurred = cv2.GaussianBlur(src=dilation, ksize=(5, 5), sigmaX=0)
        cv2.imshow('HSV1', blurred)
        cv2.waitKey(2)
        sorted_rectangles, selected_regions, letters, distances = select_roi_za_manja_slova(cv2.imread(image_path).copy(), blurred)

    #cv2.imshow('selected_regions23', selected_regions)
    #cv2.waitKey(0)

    distances = np.array(distances).reshape(len(distances), 1)

    k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
    try:
        k_means.fit(distances)
    except:
        print('ne radi k_means')

    alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                'Š', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z', 'Ž', 'a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q',
                'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']
    outputs = convert_output(alphabet)

    inputs = prepare_for_ann(letters)
    try:
        results = trained_model.predict(np.array(inputs, np.float32))
        print(display_result(results, alphabet, k_means))
    except:
        print('ne radi predict')


    try:
        extracted_text = display_result(results, alphabet, k_means)
    except:
        extracted_text = ' '
    # TODO - Izvuci tekst sa ulazne fotografije i vratiti ga kao string

    extracted_text = extracted_text.replace('Thjs', 'This')
    extracted_text = extracted_text.replace('js', 'is')

    """splitovano = extracted_text.split(" ")                         """
    """naj = 0                                                        """
    """for rec in splitovano:                                         """
    """    for key in vocabulary:                                     """
    """        x = fuzz.ratio(rec, key)                               """
    """        if x > 90:                                             """
    """            extracted_text = extracted_text.replace(rec, key)  """

    for word in extracted_text.split():
        max = 0
        new_word = ""
        for dic_word in getList(vocabulary):
            value = fuzz.ratio(word, dic_word)
            if max < value:
                max = value
                new_word = dic_word

        extracted_text = extracted_text.replace(word, new_word)

    return extracted_text
