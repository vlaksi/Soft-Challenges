from advanced_service import *
from services.postprocess import procesiraj
from services.show_result import display_result


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

    img, opening = skew_correction(image, opening)

    # POCETAK PRONALAZENJA KONTURA

    imga, contours, hierarchy = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_crtanje = img.copy()
    regions_array = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # ZA MALE REGIONE
    # print("pronadjeno kontura: " + str(len(contours)))
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        k = y - 20
        if k < 1:
            k = y
        region = opening[y:y + h + 1,
                 x:x + w + 1]  # UBACITI NEKI HENDLER, tipa ako je y-20 <0 uraditi y - 10 tako nesto
        area = cv2.contourArea(contour)
        if w < 10 or h < 45 or (h + w) < 50:
            continue
        # # ZA SVAKI REGION RADIM POBOLJSANJE
        # TODO: PROVERITI DA LI OVO RADI BOLJE !!!!
        region = cv2.morphologyEx(region.copy(), cv2.MORPH_CLOSE, kernel, iterations=1)

        regions_array.append([resize_region(region), (x, y, w, h)])

    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions_array]

    return sorted_regions


def load_image_and_find_roi_HSV_validate(image_path):
    image, img, opening = image_segmentation(image_path)
    percent_white_pixel = get_percents_for_white_and_black_pixels(img, opening)
    print("procenat belih: " + str(percent_white_pixel))

    img, opening, percent_white_pixel = check_better_channel(image, image_path, opening, percent_white_pixel)
    print("novi procenat belih: " + str(percent_white_pixel))

    if percent_white_pixel > 25:
        # img, opening, percent_white_pixel = advanced_image_segmentation(image_path)
        img, opening, percent_white_pixel = image_segmentation_in_range(image_path)

    print("najnoviji procenat belih: " + str(percent_white_pixel))

    region_distances, sorted_regions = find_roi(img, opening, percent_white_pixel)

    return region_distances, sorted_regions


def image_segmentation_in_range(image_path):
    img = cv2.imread(image_path)
    image = img.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # DEFINISEM GRANICE ZA SVE BOJE KOJE SE UGLAVNOM POJAVLJUJU
    bgr_collors = []
    yellow_bgr = (100, 220, 220)
    bgr_collors.append(yellow_bgr)

    purple_pink_bgr = (230, 182, 213)  # ne radi nesto bas sjajno
    bgr_collors.append(purple_pink_bgr)

    brown_orange_bgr = (110, 160, 220)
    bgr_collors.append(brown_orange_bgr)

    light_brown_bgr = (124, 165, 234)  # nije bas sjajno, slika 96
    bgr_collors.append(light_brown_bgr)

    # primer slike je: 46
    light_orange_bgr = (50, 170, 210)
    bgr_collors.append(light_orange_bgr)

    # primer slike je: 47
    dark_pink_bgr = (193, 170, 230)  # nije bas sjajno, slika 47
    bgr_collors.append(dark_pink_bgr)

    # primer slike je: 49
    pink_bgr = (217, 180, 245)  # ne funkcionise bas
    bgr_collors.append(pink_bgr)

    # primer slike 50

    possible_borders = []
    for bgr_collor in bgr_collors:
        lower, upper = get_limits_for_wanted_color(bgr_collor)
        possible_borders.append([lower, upper])

    # PRODJEM KROZ SVAKU GRANICU I BOJU I VIDIM KOJA IMA NAJMANJI PROCENAT
    # BELIH PIKSELA, I VEROVATNO JE ONA NAJBOLJE URADILA ODABIR SLOVA
    best_percent = 101
    best_mask = None
    u,l = None, None
    for border in possible_borders:
        temp_mask = cv2.inRange(hsv, border[0], border[1])
        percent_white_pixel = get_percents_for_white_and_black_pixels(img, temp_mask)
        # print("\t\t\t" + (str(percent_white_pixel)) + "%")
        if best_percent > percent_white_pixel > 4:
            best_percent = percent_white_pixel
            best_mask = temp_mask
            # print("novi best: " + str(best_percent))
        l,u = border[0],border[1]

    if best_mask is None:
        best_mask = cv2.inRange(hsv, l, u)
    # if best_mask is not None:
    #     plt.imshow(best_mask, 'gray')
    #     # KASNIJE MOZDA ITERIRATI I VIDETI ZA KOJU KOMBINACIJU DAJE
    #     # 'DOBRU RECENICU'
    # else:
    #     print("nije nasao koristan range")
    percent_white_pixel = get_percents_for_white_and_black_pixels(img, best_mask)
    return img, best_mask, percent_white_pixel


def advanced_image_segmentation(image_path):
    img = cv2.imread(image_path)
    image = img.copy()
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    a_component = lab[:, :, 1]
    edged = cv2.Canny(a_component, 10, 30)
    # plt.imshow(edged, 'gray')
    blur = cv2.GaussianBlur(edged, (9, 9), 0)
    # blur = cv2.medianBlur(edged,1)
    # plt.imshow(blur, 'gray')
    # print(blur.shape)
    ret, image_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
    invertovana = invert(image_bin)
    # plt.imshow(invertovana, 'gray')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(invertovana, cv2.MORPH_OPEN, kernel, iterations=3)
    # plt.imshow(opening, 'gray')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel, iterations=4)
    # plt.imshow(ero, 'gray')
    percent_white_pixel = get_percents_for_white_and_black_pixels(img, opening)
    return img, opening, percent_white_pixel


def check_better_channel(image, image_path, opening, percent_white_pixel):
    if percent_white_pixel > 20:
        image, img, opening = image_segmentation(image_path, 0)
        img, opening = skew_correction(image, opening, 0)
    else:
        img, opening = skew_correction(image, opening, 1)
    plt.imshow(opening, 'gray')
    percent_white_pixel = get_percents_for_white_and_black_pixels(img, opening)
    return img, opening, percent_white_pixel


def find_roi(img, opening, percentWhitePixel):
    imga, contours, hierarchy = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_crtanje = img.copy()
    regions_array = []
    # print("pronadjeno kontura: " + str(len(contours)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # ZA MALE REGIONE
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        k = y - 20
        if k < 1:
            k = y

        region = opening[y:y + h + 1,
                 x:x + w + 1]  # UBACITI NEKI HENDLER, tipa pako je y-20 <0 uraditi y - 10 tako nesto
        if (w < 5 + percentWhitePixel) or (h < 25 + percentWhitePixel) or (h + w) < 40:
            # print("w: " + str(w) + " h: " + str(h) + " size: " + str(h + w))
            if h > 20 + percentWhitePixel and w < 10 + percentWhitePixel:  # vrv je I u pitanju
                region = cv2.morphologyEx(region.copy(), cv2.MORPH_CLOSE, kernel, iterations=1)
                regions_array.append([resize_region(region), (x, y, w, h)])
                continue
            else:  # sum koji samo preskacemo
                continue
        region = cv2.morphologyEx(region.copy(), cv2.MORPH_CLOSE, kernel, iterations=1)
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
    return region_distances, sorted_regions


def image_segmentation(image_path, channel=1):
    # TODO: ISKORISTITI I OVO U TRAIN-U, mada ne mora jer on je okej za kanal 1
    img = cv2.imread(image_path)
    image = img.copy()
    best_channel = image[:, :, channel]
    ret, image_bin = cv2.threshold(best_channel, 0, 255, cv2.THRESH_OTSU)
    invertovana = invert(image_bin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(invertovana, cv2.MORPH_OPEN, kernel, iterations=1)
    return image, img, opening


def get_percents_for_white_and_black_pixels(img, opening):
    # BROJANJE BROJA BELIH I CRNIH PIKSELA KAKO BIH ZNAO DA LI SU SLOVA VECEG ILI MANJEG FONTA
    # PA NA OSNOVU TOGA KASNIJE RADIO ODREDJENO SKALIRANJE
    # get all non black Pixels
    numWhitePixel = cv2.countNonZero(opening)
    # print("belih: " + str(numWhitePixel))
    # get pixel count of image
    height, width, channels = img.shape
    numTotalPixel = height * width
    # print("ukupno: " + str(numTotalPixel))
    # compute all black pixels
    numBlackPixel = numTotalPixel - numWhitePixel
    # print("crnih: " + str(numBlackPixel))
    percentBlackPixel = numBlackPixel / numTotalPixel * 100
    percentWhitePixel = numWhitePixel / numTotalPixel * 100
    # print("crnih: " + str(int(percentBlackPixel)) + "%")
    # print("belih: " + str(int(percentWhitePixel)) + "%")
    return percentWhitePixel


def skew_correction(image, opening, channel=1):
    # ISPRAVLJANJE SLIKE
    # link: https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
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
    img = rotirana
    image = img.copy()
    best_channel = image[:, :, channel]
    ret, image_bin = cv2.threshold(best_channel, 0, 255, cv2.THRESH_OTSU)
    invertovana = invert(image_bin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(invertovana, cv2.MORPH_OPEN, kernel, iterations=1)
    return img, opening


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


def get_limits_for_wanted_color(wanted_bgr_color):
    wanted_color = np.uint8([[wanted_bgr_color]])  # here insert the bgr values which you want to convert to hsv
    hsv_wanted_color = cv2.cvtColor(wanted_color, cv2.COLOR_BGR2HSV)
    # print(hsv_wanted_color)

    lower_limit_for_wanted_color = hsv_wanted_color[0][0][0] - 10, 100, 100
    upper_limit_for_wanted_color = hsv_wanted_color[0][0][0] + 10, 255, 255

    return np.array(list(lower_limit_for_wanted_color), dtype="uint8"), np.array(list(upper_limit_for_wanted_color),
                                                                                 dtype="uint8")
