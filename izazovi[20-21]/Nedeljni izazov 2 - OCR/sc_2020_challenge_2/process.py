# import libraries here

from pocetak import *
from services.preparation_for_neural_network import *


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
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju

    print("train_image_paths: " + str(train_image_paths))

    model = None
    letters = None
    letters = load_image_and_find_roi_train(train_image_paths[1])
    # for train_image_path in train_image_paths:
    #     if letters is None:
    #         letters = pocetak(train_image_path)
    #     else:
    #         letters= letters + pocetak(train_image_path)

        #letters.append(letters_temp)
    print("ukupno regiona sa slovima: " + str(len(letters)))
    # alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H','I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S','Š', 'T', 'U','V', 'W', 'X', 'Y', 'Z', 'Ž']
    alphabet = ['a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q',
                'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']
    # alphabet = ALPHABET

    inputs = prepare_for_ann(letters)
    outputs = convert_output(alphabet)
    print("duzina inputa: " + str(len(inputs)))
    print("duzina outputs: " + str(len(outputs)))

    # probaj da ucitas prethodno istreniran model
    ann = load_trained_ann()

    # ako je ann=None, znaci da model nije ucitan u prethodnoj metodi i da je potrebno istrenirati novu mrezu
    if ann is None:
        print("Traniranje modela zapoceto.")
        ann = create_ann()
        ann = train_ann(ann, inputs, outputs)
        print("Treniranje modela zavrseno.")
        # serijalizuj novu mrezu nakon treniranja, da se ne trenira ponovo svaki put
        serialize_ann(ann)

    model = ann
    return model


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
    extracted_text = ""
    # TODO - Izvuci tekst sa ulazne fotografije i vratiti ga kao string
    if "train0" in image_path:
        print("img_path: " + image_path)

    distances, letters = load_image_and_find_roi_validate(image_path)

    if "train0" in image_path:
        print('Broj prepoznatih regiona:', len(letters))

    if len(letters) < 3:
        print("LOSA SEGMENTACIJA SE DESILA: pronadjeno manje od 3 slova")
    else:
        extracted_text = extract_text(distances, letters, trained_model, vocabulary)

        if "train0" in image_path:
            print(extracted_text)

    print("\n")

    return extracted_text




