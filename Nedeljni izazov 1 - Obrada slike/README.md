# Koristno


## Otso metoda
Napravi histogram i onda prolazi za razlicite vrednosti thresholda. Onda racuna ukupno rastojanje svih vrednosti
levo od tog thresholda i svih vrednosti desno od tog thresholda od njihovih srednjih vrednosti.
Uzmem threshold, izracunam srednju vrednost sa leve strane i onda gledam rastojanje svih ostalih u odnosu na tu srednju vrednost
tako i za desnu stranu. Ono sto hocemo je da smanjimo varijaciju sa obe strane thresholda.

## Histogram 
Nacin da prikazemo ili da sadrzimo info o frekvenciji odredjenih frekvencija ili odredjenih boja u nasoj slici.
Tj imamo prebrojane piksele koji su skroz crni, prebrojane piksele koji su skroz beli i prebrojane piksele koji su skroz sivi.
Tj imacemo broj piksela koji su istog osvetljenja.

Pseudo-kod histograma za grayscale sliku:

```sh
code
inicijalizovati nula vektor od 256 elemenata

za svaki piksel na slici:
  preuzeti inicijalni intezitet piksela
  uvecati za 1 broj piksela tog inteziteta
```


Kada to iscrtamo, na x osi imamo osvetljenja od 0 do 255 gde je 0 - bela & 255 - crna dok je na y osi broj pojavljivanja odredjenog piksela(osvetljenja)

#### Razmisljanje:

Ako pogledamo sliku: https://prnt.sc/vaztxs primeticemo da je generalno slika dosta tamnija nego svetlija
pa nam to moze reci da bi valjalo da je posvetlimo da bi uvideli odredjene razlike (odnosno ono sto nam je potrebno).

## Adaptivni threshold

Adaptivni threshold gde se prag racuna = tezinska suma okolnih piksela, gde su tezine iz gausove raspodele

  - https://prnt.sc/vazl1w u sredini imamo najvecu vrednost okolo se smanjuje, drugi parametar je vrednost koju postavljamo ako je ona sa kojom poredimo veca od thresholda
  - cetvrti parametar je velicina kernela[prozor, tj deo slike, npr: https://prnt.sc/vazmh0 <-- slika kernela] i mi tu velicinu odredjujemo kao na oriju, za broj slojeva sto smo odlucivali, to namestamo mozda i po nekom procentu ali nema neke konvencije i to je jako vazan parametar ovde znaci 15 x 15

U gausovom kernelu, u sredini se nalazi najveca vrednost a okolo se vrednosti smanjuju
