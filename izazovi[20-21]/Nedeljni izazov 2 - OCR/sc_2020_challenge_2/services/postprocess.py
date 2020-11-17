from fuzzywuzzy import process


def procesiraj(recenica_za_obradu,vocabulary):
    strOptions = getList(vocabulary)
    recenica = recenica_za_obradu.split(' ')
    procesirana_recenica = None
    for rec in recenica:
        print("rec: " + rec)
        str2Match = rec
        # Ratios = process.extract(str2Match, strOptions)
        # print(Ratios)
        # You can also select the string with the highest matching percentage
        highest = process.extractOne(str2Match, strOptions)
        print(highest)
        if procesirana_recenica is None:
            procesirana_recenica = highest[0] + ' '
        else:
            procesirana_recenica = procesirana_recenica + highest[0] + ' '

    # Samo da izbacim ' ' sa kraja
    procesirana_recenica = procesirana_recenica[:-1]

    return procesirana_recenica
    # str2Match = "apple inc"
    # strOptions = ["Apple Inc.","apple park","apple incorporated","iphone"]
    # Ratios = process.extract(str2Match,strOptions)
    # print(Ratios)
    # # You can also select the string with the highest matching percentage
    # highest = process.extractOne(str2Match,strOptions)


def getList(dict):
    list = []
    for key in dict.keys():
        list.append(key)

    return list