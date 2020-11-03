import numpy as np
import cv2


def count_blood_cells(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj crvenih krvnih zrnaca, belih krvnih zrnaca i
    informaciju da li pacijent ima leukemiju ili ne, na osnovu odnosa broja krvnih zrnaca

    Ova procedura se poziva automatski iz main procedure i taj deo kod nije potrebno menjati niti implementirati.

    :param image_path: <String> Putanja do ulazne fotografije.
    :return: <int>  Broj prebrojanih crvenih krvnih zrnaca,
             <int> broj prebrojanih belih krvnih zrnaca,
             <bool> da li pacijent ima leukemniju (True ili False)
    """
    red_blood_cell_count = 0
    white_blood_cell_count = 0
    has_leukemia = None

    img = cv2.imread(image_path)

    white_blood_cell_count = get_wbc(img)

    # TODO - Prebrojati crvena i bela krvna zrnca i vratiti njihov broj kao povratnu vrednost ove procedure

    # TODO - Odrediti da li na osnovu broja krvnih zrnaca pacijent ima leukemiju i vratiti True/False kao povratnu vrednost ove procedure

    return red_blood_cell_count, white_blood_cell_count, has_leukemia


def get_wbc(img):
    granica_za_wbc = 1000
    # TRAZIM MASKU KAKO BIH DOBIO SAMO WBC

    # boja za ljubicaste(odnosno bela krvna zrnca tj. WBC)
    wbc_bgr_color = [156, 71, 129]
    llfwc, ulfwc = get_limits_for_wanted_color(wbc_bgr_color)
    mask = get_mask(img, llfwc, ulfwc)

    # VRSIM OTVARANJE KAKO BIH UKLONIO SUM
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # VRSIM DILACIJU KAKO BIH UTVRDIO WBC REGIONE
    kernel = np.ones((8, 8), np.uint8)
    dilation = cv2.dilate(opening, kernel, iterations=1)

    # TRAZIM KONTURE
    _, contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    zeljene_konture = []
    for c in contours:
        # print(cv2.contourArea(c))
        if cv2.contourArea(c) > granica_za_wbc:
            zeljene_konture.append(c)

    return len(zeljene_konture)
    # draw_wanted_contours(img, zeljene_konture)


# def draw_wanted_contours(img,zeljene_konture):
#     image_crtanje = img.copy()
#     cv2.drawContours(image_crtanje, zeljene_konture, -1, (255, 0, 0), 1)
#     plt.imshow(image_crtanje)

def get_limits_for_wanted_color(wanted_bgr_color):
    wanted_color = np.uint8([[wanted_bgr_color]])  # here insert the bgr values which you want to convert to hsv
    hsv_wanted_color = cv2.cvtColor(wanted_color, cv2.COLOR_BGR2HSV)
    # print(hsv_wanted_color)

    lower_limit_for_wanted_color = hsv_wanted_color[0][0][0] - 10, 100, 100
    upper_limit_for_wanted_color = hsv_wanted_color[0][0][0] + 10, 255, 255

    return lower_limit_for_wanted_color, upper_limit_for_wanted_color

    # print(upper_limit_for_wanted_color)
    # print(lower_limit_for_wanted_color)


def get_mask(img, lower_limit_for_wanted_color, upper_limit_for_wanted_color):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_color = np.array(list(lower_limit_for_wanted_color))
    upper_color = np.array(list(upper_limit_for_wanted_color))
    mask = cv2.inRange(img_hsv, lower_color, upper_color)
    return mask
    # plt.imshow(mask)
