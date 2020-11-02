# import libraries here
import cv2
import numpy as np

def count_blood_cells(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj krvnih zrnaca.

    Ova procedura se poziva automatski iz main procedure i taj deo kod nije potrebno menjati niti implementirati.

    :param image_path: <String> Putanja do ulazne fotografije.
    :return: <int>  Broj prebrojanih krvnih zrnaca
    """
    # TODO - Prebrojati krvna zrnca i vratiti njihov broj kao povratnu vrednost ove procedure
    img = cv2.imread(image_path)

    param_kernel_morph = (5, 5)
    param_kernel_deliate = (8, 8)
    param_min_area = 3800
    param_max_area = 15900
    img_hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 15, 5])
    upper_red = np.array([20, 50, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    lower_red = np.array([150, 15, 5])
    upper_red = np.array([180, 50, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    #cv2.imshow("mask0", mask0)
    #cv2.imshow("mask1", mask1)
    mask = mask0 + mask1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, param_kernel_morph)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel = np.ones(param_kernel_deliate, np.uint8)
    dilation = cv2.dilate(opening, kernel, iterations=1)
    a, cnts, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    bonus = 0
    for c in cnts:
        if (cv2.contourArea(c) > param_min_area):
            # if(cv2.contourArea(c)>15000 and cv2.contourArea(c)<20000):
            #     continue
            if (cv2.contourArea(c) > param_max_area):
                bonus += 1
                pass
            rect = cv2.boundingRect(c)
            rects.append(rect)
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
            pass
        # elif(cv2.contourArea(c)>2000):
        #     rect = cv2.boundingRect(c)
        #     cv2.rectangle(img,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,255,0),2)
        #     pass
        else:
            pass
        pass
    print(len(rects)+bonus)
    #cv2.imshow("mask", mask)
    cv2.imshow('image2', img)
    cv2.waitKey(0)

    return len(rects)+bonus