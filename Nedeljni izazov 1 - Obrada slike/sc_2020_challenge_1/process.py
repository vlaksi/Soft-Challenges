import numpy as np
import cv2

"""
Ideju koju trenutno imam je da uzmem prvo nadjem crvene, pa onda ove white_blood
za svaku vrstu prebrojim i to je to.

Potom logiku da odlucim na osnovu tih odnosa da li je u pitanju ili nije leukemija.

Jako bitno za izazov-1

https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html
"""


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

    # We start with finding an approximate estimate of the coins. For that, we can use the Otsu's binarization.
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # plt.imshow(img)

    # To remove any small holes in the object, we can use morphological closing.
    # So, now we know for sure that region near to center of objects are foreground
    # and region much away from the object are background.

    # zatvaranje = dilacija + erozija, zatvaranje sitnih otvora među belim pikselima
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # img_dil = cv2.dilate(thresh, kernel, iterations=5)
    # img_close = cv2.erode(img_dil, kernel, iterations=1)
    # plt.imshow(img_close, 'gray')
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # plt.imshow(sure_fg, 'gray')

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    #plt.imshow(markers)


    #red_blood_cell_count = pronalazenje_kontura(image_path)

    #loaded_image = image_processing(image_path)

    # image_segmentation(loaded_image)

    # TODO - Prebrojati crvena i bela krvna zrnca i vratiti njihov broj kao povratnu vrednost ove procedure
    # print("\n\n\t\t image_path\n\n" + loadedImage.shape + "\n\n")
    # print(loaded_image)

    # TODO - Odrediti da li na osnovu broja krvnih zrnaca pacijent ima leukemiju i vratiti True/False kao povratnu vrednost ove procedure

    return red_blood_cell_count, white_blood_cell_count, has_leukemia


def pronalazenje_kontura(image_path):
    # ucitavanje slike i convert u RGB
    img_barcode = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # konvertovanje u grayscale i adaptivni threshold.
    img_barcode_gs = cv2.cvtColor(img_barcode, cv2.COLOR_RGB2GRAY)  # konvert u grayscale

    # ret, image_barcode_bin = cv2.threshold(img_barcode_gs, 80, 255, cv2.THRESH_BINARY)
    image_barcode_bin = cv2.adaptiveThreshold(img_barcode_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 10)

    # pronalazenje kontura/regiona
    img, contours, hierarchy = cv2.findContours(image_barcode_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img = img_barcode.copy()
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    #plt.imshow(img)

    return 1


def image_processing(image_path):
    loaded_image = cv2.imread(image_path)
    loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
    # TODO: Change to HSV for better processing in image segmentation
    # loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
    return loaded_image


def image_processing_with_ada_threshold(image_path):
    """
    adaptivni threshold gde se prag racuna = tezinska suma okolnih piksela, gde su tezine iz gausove raspodele
    https://prnt.sc/vazl1w u sredini imamo najvecu vrednost okolo se smanjuje
    drugi parametar je vrednost koju postavljamo ako je ona sa kojom poredimo veca od thresholda
    cetvrti parametar je velicina kernela[prozor, tj deo slike, npr: https://prnt.sc/vazmh0 <-- slika kernela] i mi tu velicinu odredjujemo
    kao na oriju, za broj slojeva sto smo odlucivali, to namestamo mozda i po nekom procentu ali nema neke konvencije
    i to je jako vazan parametar ovde znaci 15 x 15
    U gausovom kernelu, u sredini se nalazi najveca vrednost a okolo se vrednosti smanjuju
    """

    loaded_image_ada = cv2.imread(image_path)
    loaded_image_ada = cv2.cvtColor(loaded_image_ada, cv2.COLOR_BGR2GRAY)
    image_ada_bin = cv2.adaptiveThreshold(loaded_image_ada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15,
                                          5)

    return image_ada_bin


def blob_detection():
    print('blob detection')


def cell_counting():
    """
    Trenutno mislim da bih mogao iskoristiti histogram. Imaju veze sa otso metodom.

    Sta radi Otso:
    Napravi histogram i onda prolazi za razlicite vrednosti thresholda. Onda racuna ukupno rastojanje svih vrednosti
    levo od tog thresholda i svih vrednosti desno od tog thresholda od njihovih srednjih vrednosti.
    Uzmem threshold, izracunam srednju vrednost sa leve strane i onda gledam rastojanje svih ostalih u odnosu na tu srednju vrednost
    tako i za desnu stranu. Ono sto hocemo je da smanjimo varijaciju sa obe strane thresholda.

    Histogram: nacin da prikazemo ili da sadrzimo info o frekvenciji odredjenih frekvencija ili odredjenih boja u nasoj slici.
    Tj imamo prebrojane piksele koji su skroz crni, prebrojane piksele koji su skroz beli i prebrojane piksele koji su skroz sivi.
    Tj imacemo broj piksela koji su istog osvetljenja.
    :return:
    """
    print('sell counting')


def image_segmentation(loaded_image):
    # print(loaded_image.shape)
    # Current problem why i not use HSV is because we get something like this: https://prnt.sc/vazewf
    # and that is not suitable for cv2.threshold() function
    # she expect something like this: https://prnt.sc/vazf3d
    ret, image_bin = cv2.threshold(loaded_image, 0, 255,
                                   cv2.THRESH_OTSU)  # ret je izracunata vrednost praga, image_bin je binarna slika
    print("Otsu's threshold: " + str(ret))


def histogram(image):
    """

    MOZE SE KORISTITI I OD OpenCV-a:
    hist_full = cv2.calcHist([img_gray], [0], None, [255], [0, 255])

    Pseudo-kod histograma za grayscale sliku:

    code
    inicijalizovati nula vektor od 256 elemenata

    za svaki piksel na slici:
        preuzeti inicijalni intezitet piksela
        uvecati za 1 broj piksela tog inteziteta


    Kada to iscrtamo, na x osi imamo osvetljenja od 0 do 255 gde je 0 - bela & 255 - crna
    dok je na y osi broj pojavljivanja odredjenog piksela(osvetljenja)

    Razmisljanje:
    Ako pogledamo sliku: https://prnt.sc/vaztxs primeticemo da je generalno slika dosta tamnija nego svetlija
    pa nam to moze reci da bi valjalo da je posvetlimo da bi uvideli odredjene razlike (odnosno ono sto nam je potrebno).

    :param image:
    :return:
    """
    height, width = image.shape[0:2]
    x = range(0, 256)
    y = np.zeros(256)

    for i in range(0, height):
        for j in range(0, width):
            pixel = image[i, j]
            y[pixel] += 1

    return (x, y)
