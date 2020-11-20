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


def load_image_and_find_roi_HSV_TRAIN(image_path):
    img = cv2.imread(image_path)

    image = img.copy()
    best_channel = image[:, :, 1]
    ret, image_bin = cv2.threshold(best_channel, 0, 255, cv2.THRESH_OTSU)
    invertovana = invert(image_bin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(invertovana, cv2.MORPH_OPEN, kernel, iterations=1)

    # ISPRAVLJANJE SLIKE
    coords = np.column_stack(np.where(opening > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # print("ugao: " + str(angle))
    if angle < - 45:
        angle = - (90 + angle)
    else:
        angle = -angle
    # print(angle)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotirana = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    plt.imshow(rotirana, 'gray')

    img = rotirana
    image = img.copy()
    best_channel = image[:, :, 1]
    ret, image_bin = cv2.threshold(best_channel, 0, 255, cv2.THRESH_OTSU)
    invertovana = invert(image_bin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(invertovana, cv2.MORPH_OPEN, kernel, iterations=1)
    # plt.imshow(opening, 'gray')

    # ZAVRSENO ISPRAVLJANJE SLIKE

    # POCETAK PRONALAZENJA KONTURA

    imga, contours, hierarchy = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_crtanje = img.copy()
    regions_array = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # ZA MALE REGIONE
    # print("pronadjeno kontura: " + str(len(contours)))
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        region = opening[y - 20:y + h + 1,
                 x:x + w + 1]  # UBACITI NEKI HENDLER, tipa ako je y-20 <0 uraditi y - 10 tako nesto
        area = cv2.contourArea(contour)
        if w < 10 or h < 45 or (h + w) < 50:
            continue
        # # ZA SVAKI REGION RADIM POBOLJSANJE
        # TODO: PROVERITI DA LI OVO RADI BOLJE !!!!
        # region = cv2.morphologyEx(region.copy(), cv2.MORPH_DILATE, kernel, iterations=3)

        regions_array.append([resize_region(region), (x, y, w, h)])

    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions_array]

    return sorted_regions


def load_image_and_find_roi_validate(image_path):
    # Učitavanje slike i određivanje regiona od interesa
    image_color = load_image(image_path)
    img = invert(image_bin_optimized(image_optimized_channel(image_color)))
    # noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    # ciscenje slova
    erozija = cv2.erode(sure_bg.copy(), kernel, iterations=3)
    img_bin = erozija.copy()
    #     plt.imshow(img_bin, 'gray')
    selected_regions, letters, distances = select_roi(image_color.copy(), img_bin)
    # display_image(selected_regions)
    return distances, letters


def load_image_and_find_roi_HSV_validate(image_path):
    img = cv2.imread(image_path)

    image = img.copy()
    best_channel = image[:, :, 1]
    ret, image_bin = cv2.threshold(best_channel, 0, 255, cv2.THRESH_OTSU)
    invertovana = invert(image_bin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(invertovana, cv2.MORPH_OPEN, kernel, iterations=1)

    # ISPRAVLJANJE SLIKE
    coords = np.column_stack(np.where(opening > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # print("ugao: " + str(angle))
    if angle < - 45:
        angle = - (90 + angle)
    else:
        angle = -angle
    # print(angle)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotirana = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    plt.imshow(rotirana, 'gray')

    img = rotirana
    image = img.copy()
    best_channel = image[:, :, 1]
    ret, image_bin = cv2.threshold(best_channel, 0, 255, cv2.THRESH_OTSU)
    invertovana = invert(image_bin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(invertovana, cv2.MORPH_OPEN, kernel, iterations=1)
    # plt.imshow(opening, 'gray')

    # ZAVRSENO ISPRAVLJANJE SLIKE

    # POCETAK PRONALAZENJA KONTURA

    imga, contours, hierarchy = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_crtanje = img.copy()
    regions_array = []
    # print("pronadjeno kontura: " + str(len(contours)))
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        k = y - 20
        if k < 1:
            k = y

        region = opening[k:y + h + 1,
                 x:x + w + 1]  # UBACITI NEKI HENDLER, tipa ako je y-20 <0 uraditi y - 10 tako nesto
        if w < 10 or h < 45 or (h + w) < 50:
            continue
        regions_array.append([resize_region(region), (x, y, w, h)])

    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])  # X_next - (X_current + W_current)
        region_distances.append(distance)

    # display_image(image_crtanje)
    # plt.imshow(image_crtanje)

    return region_distances, sorted_regions


def extract_text(distances, letters, trained_model, vocabulary):
    # Podešavanje centara grupa K-means algoritmom
    distances = np.array(distances).reshape(len(distances), 1)
    # Neophodno je da u K-means algoritam bude prosleđena matrica u kojoj vrste određuju elemente
    k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=20)
    k_means.fit(distances)
    ## PREDIKCIJA
    alphabet0 = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                 'S', 'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž']
    alphabet1 = ['a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                 'n', 'o', 'p', 'q',
                 'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']
    alphabet = alphabet0 + alphabet1
    inputs = prepare_for_ann(letters)
    results = trained_model.predict(np.array(inputs, np.float32))
    extracted_text = display_result(results, alphabet, k_means)
    extracted_text = procesiraj(extracted_text, vocabulary)
    return extracted_text


def extract_text_without_vocabulary(distances, letters, trained_model):
    # Podešavanje centara grupa K-means algoritmom
    distances = np.array(distances).reshape(len(distances), 1)
    # Neophodno je da u K-means algoritam bude prosleđena matrica u kojoj vrste određuju elemente
    k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=20)
    k_means.fit(distances)
    ## PREDIKCIJA
    alphabet0 = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                 'S', 'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž']
    alphabet1 = ['a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                 'n', 'o', 'p', 'q',
                 'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']
    alphabet = alphabet0 + alphabet1

    inputs = prepare_for_ann(letters)
    results = trained_model.predict(np.array(inputs, np.float32))
    extracted_text = display_result(results, alphabet, k_means)
    return extracted_text


def get_alphabet_and_letters(train_image_paths):
    # image_path0 = 'dataset/train/alphabet0.png'
    letters0 = load_image_and_find_roi_HSV_TRAIN(train_image_paths[0])
    alphabet0 = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                 'S', 'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž']
    # image_path1 = 'dataset/train/alphabet1.png'
    letters1 = load_image_and_find_roi_HSV_TRAIN(train_image_paths[1])
    alphabet1 = ['a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                 'n', 'o', 'p', 'q',
                 'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']
    alphabet = alphabet0 + alphabet1
    letters = letters0 + letters1
    return alphabet, letters
