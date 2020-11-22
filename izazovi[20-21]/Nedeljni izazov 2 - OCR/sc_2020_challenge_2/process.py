# import libraries here

from pocetak import *
from services.preparation_for_neural_network import *

BROJAC = 0

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
    alphabet, letters = get_alphabet_and_letters(train_image_paths)
    print("ukupno regiona sa slovima: " + str(len(letters)))

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
    print("\n\t--------------------POCETAK-----------------")
    print("\tslika: " + image_path)
    extracted_text = ""
    # TODO - Izvuci tekst sa ulazne fotografije i vratiti ga kao string

    distances, letters = load_image_and_find_roi_HSV_validate(image_path)

    if len(letters) < 3:
        print(image_path+" - LOSA SEGMENTACIJA SE DESILA: pronadjeno manje od 3 slova")
    else:
        extracted_text = extract_text(distances, letters, trained_model, vocabulary)
        # extracted_text = extract_text_without_vocabulary(distances, letters, trained_model)
    print("\t--------------------KRAJ-----------------\n")
    return extracted_text




