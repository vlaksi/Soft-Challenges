"""
Module for printing:
  - train or load model data
  - extract text from image data
"""


def print_for_train_or_load_model(serialization_folder, train_image_paths):
    print("train_image_paths: " + str(train_image_paths))
    print("serialization_folder: " + serialization_folder)
    print("\n")


def print_for_extract_text_from_image(image_path, trained_model, vocabulary):
    print("trained_model: " + str(trained_model))
    print("image_path: " + image_path)
    print("vocabulary: " + str(vocabulary))