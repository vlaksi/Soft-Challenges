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
    red_blood_cell_count = get_rbc_with_watershed_optimized(image_path)

    print("path: " + image_path)
    print("rbc: " + str(red_blood_cell_count))
    print("wbc: " + str(white_blood_cell_count))

    ukupno = red_blood_cell_count + white_blood_cell_count
    procenat = (white_blood_cell_count / ukupno) * 100
    print("procenat wbc: " + str(procenat) + "\n")

    # EDGE CASEOVI ZA SLUCAJ DA MI NE RADI ALGORITAM
    # if red_blood_cell_count <= 4:
    #     red_blood_cell_count = white_blood_cell_count * 5

    # TODO: NADJI PAMETNIJU PROVERU DA LI IMA ILI NEMA LEUKEMIJU
    if procenat > 8.0:
        has_leukemia = True
    else:
        has_leukemia = False

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


def get_rbc(img_path):
    """
    OVO MI SE CINI OKEJ
    :param img_path:
    :return:
    """
    # read original image
    image = cv2.imread(img_path)
    # img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    # plt.imshow(image)

    # MENJAM KANAL ZA RBC
    img = image[:, :, 0]
    # plt.imshow(img)

    # THRESHOLDUJEM DA BIH MOGAO LAKSE DA IZDVAJAM KONTURE
    ret, bin_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_im = 255 - bin_im
    # plt.imshow(binIm, 'gray')

    # drugaciji strukturni element
    # RADIM DILACIJU DA UTEMELJIM BELE KONTURE/CELIJE/POVRSINE KOJE SE RACUNAJU
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # MORPH_ELIPSE, MORPH_RECT...
    dilate_img = cv2.dilate(bin_im, kernel, iterations=1)
    # plt.imshow(dilate_img, 'gray')  # 5 iteracija

    # PRONALAZIM BELE KONTURE
    _, cnts, _ = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # image_crtanje = image.copy()
    # cv2.drawContours(image_crtanje, cnts, -1, (255, 0, 0), 1)
    # plt.imshow(image_crtanje)

    zeljene_konture = []
    for c in cnts:
        # print(cv2.contourArea(c))
        if 150 < cv2.contourArea(c) < 10000:
            # print("dodao")
            # print(cv2.contourArea(c))
            zeljene_konture.append(c)

    # image_crtanje = image.copy()
    # cv2.drawContours(image_crtanje, zeljene_konture, -1, (255, 0, 0), 1)
    # plt.imshow(image_crtanje)
    return len(zeljene_konture)


def get_rbc_watershed(img_path):
    image = cv2.imread(img_path)

    # MENJAM U KANAL POGODNIJI ZA DETEKTOVANJE RBC
    img = image[:, :, 0]
    # plt.imshow(img)

    # TRESUJEM DA DOBIJEM SAMO BELE
    ret, bin_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_im = 255 - bin_im  # OBRCEM KAKO BIH KASNIJE BROJAO BELE (ili crne, nisam sig :D )
    # plt.imshow(bin_im, 'gray')

    # ODREDJUJEM STA JE SIGURNO BACKGROUND
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # MORPH_ELIPSE, MORPH_RECT...
    sure_bg = cv2.dilate(bin_im, kernel, iterations=4)
    # plt.imshow(sure_bg, 'gray')  # 5 iteracija

    # ODREDJUJEM STA JE SIGURNO FOREGROUND
    dist_transform = cv2.distanceTransform(bin_im, cv2.DIST_L2, 3)
    ret2, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    # plt.imshow(sure_fg, 'gray')

    # POKUSAVAM DA GRANICE RESIM NEKAKO - smanjivanjem broja belikih piksela
    granica_sure_fg = cv2.erode(sure_fg, (3, 3), iterations=3)
    # plt.imshow(granica_sure_fg, 'gray')

    # NEPOZNATA REGIJA
    unknown = cv2.subtract(sure_bg, sure_fg)

    # PRETVARANJE U MARKERE
    ret3, markers = cv2.connectedComponents(sure_fg)
    # MARKIRAMO SIGURNU POZADINU
    markers = markers + 10
    # MARKIRAMO NEPOZNATU TERITORIJU
    markers[unknown == 255] = 0
    # plt.imshow(markers)

    # PRIMENJUJEMO WATERSHED
    markers = cv2.watershed(image, markers)
    # plt.imshow(markers)

    # GRANICE SU OZNACENE SA -1 PA IM DODELJUJEM BOJU
    img1 = image.copy()
    # print(markers)
    img1[markers > 10] = [0, 0, 0]
    img1[markers <= 10] = [255, 255, 255]
    # img1[markers == -1] = [0,0,0]

    # plt.imshow(img1)

    # PRETVARAM IZ WATERSHEDA U KORISNU SLIKU ZA TRAZENJE KONTURA
    img = cv2.cvtColor(img1.copy(), cv2.COLOR_BGR2GRAY)
    ret, image_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)  # ret je vrednost praga, image_bin je binarna slika
    # plt.imshow(image_bin, 'gray')
    dilacija = cv2.dilate(image_bin, (3, 3), iterations=5)
    dilacija = 255 - dilacija
    # plt.imshow(dilacija, 'gray')

    # *** KONACNO TRAZIM KONTURE***
    # PRONALAZIM BELE KONTURE
    _, cnts, _ = cv2.findContours(dilacija, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # image_crtanje = image.copy()
    # cv2.drawContours(image_crtanje, cnts, -1, (255, 0, 0), 1)
    # plt.imshow(image_crtanje)
    # print("ukupno kontura: " + str(len(cnts)))

    # OVO ISPOD KORISTIM
    # PRONALAZIM BELE KONTURE
    _, cnts, _ = cv2.findContours(dilacija, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    zeljene_konture = []
    for c in cnts:
        if 200 < cv2.contourArea(c):
            zeljene_konture.append(c)

    # image_crtanje = image.copy()
    # cv2.drawContours(image_crtanje, zeljene_konture, -1, (255, 0, 0), 2)
    # plt.imshow(image_crtanje)
    print("zeljenih kontura: " + str(len(zeljene_konture)))
    return len(zeljene_konture)


def get_rbc_with_watershed_low_quality_image(img_path):
    """
    Trenutno zadovoljavajuce broji RBC za slike losijeg kvaliteta
    ali i za one sa manje WBC, tako mi bar deluje.
    :param img_path:
    :return:
    """

    # UCITAVAM SLIKU I THRESHODUJEM
    img = cv2.imread(img_path)
    image = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # plt.imshow(thresh, 'gray')

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    # plt.imshow(opening, 'gray')

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # plt.imshow(sure_bg, 'gray')

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    # plt.imshow(sure_fg,'gray')

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 10
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    # plt.imshow(markers)

    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]

    # GRANICE SU OZNACENE SA -1 PA IM DODELJUJEM BOJU
    img1 = img.copy()
    # print(markers)
    img1[markers > 10] = [0, 0, 0]
    img1[markers <= 10] = [255, 255, 255]
    # img1[markers == -1] = [0,0,0]
    # plt.imshow(img1)

    # PRETVARAM IZ WATERSHEDA U KORISNU SLIKU ZA TRAZENJE KONTURA
    img = cv2.cvtColor(img1.copy(), cv2.COLOR_BGR2GRAY)
    ret, image_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)  # ret je vrednost praga, image_bin je binarna slika
    image_bin = image_bin - 255
    # plt.imshow(image_bin, 'gray')

    # PRONALAZIM BELE KONTURE
    _, cnts, _ = cv2.findContours(image_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("ukupno kontura: " + str(len(cnts)))

    image_crtanje = image.copy()
    cv2.drawContours(image_crtanje, cnts, -1, (255, 0, 0), 2)
    # plt.imshow(image_crtanje)

    return len(cnts)


def get_rbc_with_watershed_optimized(img_path):
    img = cv2.imread(img_path)
    image = img.copy()

    # MENJAM U KANAL POGODNIJI ZA DETEKTOVANJE RBC
    gray = image[:, :, 0]
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # plt.imshow(thresh, 'gray')

    # noise removal
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # plt.imshow(opening, 'gray')

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # plt.imshow(sure_bg, 'gray')

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
    ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
    # plt.imshow(sure_fg,'gray')

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 10
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    # plt.imshow(markers)

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    # GRANICE SU OZNACENE SA -1 PA IM DODELJUJEM BOJU
    img1 = img.copy()
    # print(markers)
    img1[markers > 10] = [0, 0, 0]
    img1[markers <= 10] = [255, 255, 255]
    # img1[markers == -1] = [0,0,0]
    # plt.imshow(img1)

    # PRETVARAM IZ WATERSHEDA U KORISNU SLIKU ZA TRAZENJE KONTURA
    img = cv2.cvtColor(img1.copy(), cv2.COLOR_BGR2GRAY)
    ret, image_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)  # ret je vrednost praga, image_bin je binarna slika
    image_bin = image_bin - 255
    # plt.imshow(image_bin, 'gray')

    # PRONALAZIM BELE KONTURE
    _, cnts, _ = cv2.findContours(image_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("ukupno kontura: " + str(len(cnts)))

    image_crtanje = image.copy()
    cv2.drawContours(image_crtanje, cnts, -1, (255, 0, 0), 2)
    # plt.imshow(image_crtanje)

    return len(cnts)

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
