# keras
from keras.models import model_from_json

"""
Module for:
  - check is model trained
  - check is model in folder
  - train model
  - load model from folder
  - save model to folder
"""


def serialize_ann(ann):
    # serijalizuj arhitekturu neuronske mreze u JSON fajl
    model_json = ann.to_json()
    with open("serialization_folder/neuronska.json", "w") as json_file:
        json_file.write(model_json)
    # serijalizuj tezine u HDF5 fajl
    ann.save_weights("serialization_folder/neuronska.h5")


def load_trained_ann():
    try:
        # Ucitaj JSON i kreiraj arhitekturu neuronske mreze na osnovu njega
        json_file = open('serialization_folder/neuronska.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ann = model_from_json(loaded_model_json)
        # ucitaj tezine u prethodno kreirani model
        ann.load_weights("serialization_folder/neuronska.h5")
        print("Istrenirani model uspesno ucitan.")
        return ann
    except Exception as e:
        print("\n\n\n\n model prethodno nije serijalizovan pa nema odakle da bude ucitan \n\n\n\n")
        # ako ucitavanje nije uspelo, verovatno model prethodno nije serijalizovan pa nema odakle da bude ucitan
        return None
