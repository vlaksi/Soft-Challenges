from advanced_service import *
from services.postprocess import procesiraj
from services.show_result import display_result


def load_image_and_find_roi_train(path_img):
    # POCETAAAAAAAAAAAAK
    image_color = load_image(path_img)
    img = invert(image_bin_optimized(image_optimized_channel(image_color)))

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=4)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    # ciscenje slova
    erozija = cv2.erode(sure_bg.copy(), (3, 3), iterations=3)
    img_bin = erozija.copy()
    #     plt.imshow(img_bin, 'gray')
    selected_regions, letters, region_distances = select_roi(image_color.copy(), img_bin)
    display_image(selected_regions)
    print('Broj prepoznatih regiona:', len(letters))

    return letters


def load_image_and_find_roi_validate(image_path):
    # Učitavanje slike i određivanje regiona od interesa
    image_color = load_image(image_path)
    img = invert(image_bin_optimized(image_optimized_channel(image_color)))
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=4)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    # ciscenje slova
    erozija = cv2.erode(sure_bg.copy(), (3, 3), iterations=3)
    img_bin = erozija.copy()
    #     plt.imshow(img_bin, 'gray')
    selected_regions, letters, distances = select_roi(image_color.copy(), img)
    display_image(selected_regions)
    return distances, letters


def extract_text(distances, letters, trained_model,vocabulary ):
    # Podešavanje centara grupa K-means algoritmom
    distances = np.array(distances).reshape(len(distances), 1)
    # Neophodno je da u K-means algoritam bude prosleđena matrica u kojoj vrste određuju elemente
    k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=20)
    k_means.fit(distances)
    ## PREDIKCIJA
    alphabet = ['a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q',
                'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']
    inputs = prepare_for_ann(letters)
    results = trained_model.predict(np.array(inputs, np.float32))
    extracted_text = display_result(results, alphabet, k_means)
    extracted_text = procesiraj(extracted_text,vocabulary)
    return extracted_text