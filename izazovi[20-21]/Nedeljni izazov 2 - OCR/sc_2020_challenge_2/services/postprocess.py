from fuzzywuzzy import process
from fuzzywuzzy import fuzz

def procesiraj(recenica_za_obradu, vocabulary):
    """
    Koriscen fuzzy wuzzy koji je dat u linku na vezbama:
    https://www.datacamp.com/community/tutorials/fuzzy-string-python
    :param recenica_za_obradu:
    :param vocabulary:
    :return:
    """
    strOptions = getList(vocabulary)
    recenica = recenica_za_obradu.split(' ')
    procesirana_recenica = None

    for rec in recenica:
        str2Match = rec
        Ratios = process.extract(str2Match, strOptions, scorer=fuzz.token_set_ratio)
        # print(Ratios)
        highest = find_most_similar(Ratios, rec)
        # You can also select the string with the highest matching percentage
        # highest = process.extractOne(str2Match, strOptions, scorer=fuzz.token_set_ratio)
        # highest = highest[0]
        # print("\t\trec: " + rec)
        # print("\t\tnaslicnija rec: "+highest + "\n")
        if procesirana_recenica is None:
            procesirana_recenica = highest + ' '
        else:
            procesirana_recenica = procesirana_recenica + highest + ' '

    # Samo da izbacim ' ' sa kraja
    procesirana_recenica = procesirana_recenica[:-1]
    # print("\t\t recenica: " + recenica_za_obradu)
    # print("\t\t procesirana recenica: " + procesirana_recenica + "\n")

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


def find_most_similar(Ratios, word):
    """
    pokusavam da nadjem onaj najslicniji reci, ako nadjem isti kao rec odma vratim
    inace vratim prvi iz Ratios - jer on ima najveci procenat poklapanja
    """
    if len(Ratios) < 2:
        return Ratios[0][0]
    else:
        for i in range(len(Ratios)):
            if word == Ratios[i][0]:
                # [i] kako bih pristupio pravom paru ('rec', procenat), a drugi [0] kako bih pristupio 'rec'i
                return word

    return Ratios[0][0]
