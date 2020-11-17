from advanced_service import *


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
    # display_image(selected_regions)
    return distances, letters
