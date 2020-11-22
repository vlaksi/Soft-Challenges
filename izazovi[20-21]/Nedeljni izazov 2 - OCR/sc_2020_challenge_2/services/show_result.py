"""
Module for:
  - finding winner
  - display result
"""


def winner(output):  # output je vektor sa izlaza neuronske mreze
    """pronaći i vratiti indeks neurona koji je najviše pobuđen"""
    return max(enumerate(output), key=lambda x: x[1])[0]


def display_result(outputs, alphabet, k_means):
    '''
    Funkcija određuje koja od grupa predstavlja razmak između reči, a koja između slova, i na osnovu
    toga formira string od elemenata pronađenih sa slike.
    Args:
        outputs: niz izlaza iz neuronske mreže.
        alphabet: niz karaktera koje je potrebno prepoznati
        kmeans: obučen kmeans objekat
    Return:
        Vraća formatiran string
    '''
    # Odrediti indeks grupe koja odgovara rastojanju između reči, pomoću vrednosti iz k_means.cluster_centers_
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:, :]):
        # Iterativno dodavati prepoznate elemente kao u vežbi 2, alphabet[winner(output)]
        # Dodati space karakter u slučaju da odgovarajuće rastojanje između dva slova odgovara razmaku između reči.
        # U ovu svrhu, koristiti atribut niz k_means.labels_ koji sadrži sortirana rastojanja između susednih slova.

        # ako je u pitanju kvacica
        # i ako je prethodno u pitanju C, to znaci da imamo Cv
        # mi trebamo da zamenimo u C tvrdo C,
        # u trenutku kad ima Cv, on ima i razmak, pa ce ga dodati
        # tkd da treba da preskocim i ovaj if sa razmakom i samo uradim apendovanje
        # slova C(tvrdog C)

        # ako imamo Sv prebacimo u S
        letter = alphabet[winner(output)]

        if k_means.labels_[idx] == w_space_group:
            result += ' '
        result += letter
    result.replace('sv ', 'AAAA')
    return result
