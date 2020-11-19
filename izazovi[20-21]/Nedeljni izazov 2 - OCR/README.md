# Koristno

<details>
  <summary> Kvacice </summary> <br>

## Kvacice

Voditi racuna o **kvacicama(ć,č,i,j,ž )**, to treba da hendlujemo u procesiranju. Odnosno u trenutku kada isecamo slicicu, treba da proverimo da li mozda postoji neki region koji se prekla sa regionom C recimo odnosno da li se kvacica preklapa sa slovom.

Pa ako postoji drugi region koji je recimo IZNAD i nalazi se u nekim granicama , treba ta dva da spojimo u jedan region.

To resavamo tako sto nadjemo minimalni_y, maksimalni_y, min_x, max_x od ta dva regiona i na kraju nadjemo border za sve to i to ce nam biti jedan region sve zajedno.  

Voditi racuna da ne mora kvacica uvek biti u granicama leve i desne granice slova.

</details>

<details>
  <summary> Normalizacija svih inputa </summary> <br>

## Normalizacija svih inputa

Kao na slici [normalizacija-svih-inputa.png](https://prnt.sc/vjj63l) , mi cemo raditi normalizaciju svih inputa u opseg od 0-1, posto je nasa slika u opsegu od 0-255, sve sto je potrebno je da izvrsimo **deljenje sa 255** kako bi spustili na skalu od 0-1.

Voditi racuna da na ulaz mreze dodju konture koje su samo slova, jer u suprotnom moze doci do pucanja programa. Tipa ako ocekujemo da dodju slova a dodje tacka, to ce napraviti problem.

</details>

<details>
  <summary> Dimenzije </summary> <br>

## Dimenzije 

Dimenzije ulaza(broj primeraka) i izlaza moraju biti iste ! [dimenzije-ulaza-izlaza.png](https://prnt.sc/vjj6lw) 

  - Na ulazu je niz vektora(dimenzije 784), gde je svaki element(vektor) zapravo kontura oko slova, koja je iz matrice 28x28 prebacena u vektor(dimenzije 784)
  - Na izlazu samo kazemo koje slovo je aktivirano ( one dot rule)

</details>

<details>
  <summary> Kreiranje i treniranje modela </summary> <br>

## Kreiranje i treniranje modela

Valjalo bi da nam data set bude izmesan [izmesan-data-set.png](https://prnt.sc/vjj717) 

  - odnosno, shuffle da bude na true
  - jer ako nije, imamo onda sortiranu situaciju, gde recimo dodje 10 A, pa 10 B, itd itd, a kako bi bolje trenirali mrezu, potrebno je uraditi shuffle


U ovoj slici **batch_size** je parametar koji nam govori posle koliko iteracija da se radi azuriranje tezina

  - valjalo bi da je on malo vise, jer dobro bi bilo da recimo tek posle 10 iteracija uradimo azuriranje, kako bi ipak dobili neku usrednjenu vrednost


**Verbose** je parametar koji nam govori samo gde da radimo logovanje

  - 0: logovanja bez
  - 1: ekran
  - 2: fajl 

</details>

<details>
  <summary> Prikaz rezultata </summary> <br>

## Prikaz rezultata

Kod [prikaz-rezultata.png](https://prnt.sc/vjj7v1)  

  - outputs: skup verovatnoca koje dolaze iz neuronske mreze(svaka od njih u intervalu od 0 do 1) 
  - za svaku verovatnocu pokusavamo da nadjemo ko je pobednik(koje slovo ima najvecu verovatnocu)
  
</details>

<details>
  <summary> Razmaci izmedju reci </summary> <br>

## Razmaci izmedju reci

Da bi resili **razmake izmedju reci**, treba da istreniramo **k-means sa 2 klustera**, jedan za vece i jedan za manje. Na taj nacin smo resili [manje-vece-razmake.png](https://prnt.sc/vjj87d) 

Ako je recimo A blizu klasteru '**manji razmak**' onda ga samo nalepimo na prethodne karaktere. A ako je A blizi klasteru '**veci razmak**' onda je on pocetak nove reci.
  
</details>

<details>
  <summary> Vise redova </summary> <br>

## Vise redova

Klasterujemo po Y:
  
  - mozemo gledati [gornji levi ugao](https://prnt.sc/viw9q8) ili [sredinu klastera](https://prnt.sc/viwa7e) (kako god) 
  - bitno je da svi oni oko y biti oko 160 a ovi iz drugog reda ce biti oko y 240 recimo 
  - u tom slucaju smo podelili dva reda 
  - onda u prvom redu sve sortiramo po X a onda i u drugom redu sve po X, **odvojeno**

</details>

<details>
  <summary> Zarotirani text </summary> <br>

## Zarotirani text

Ako imamo ovakav [primer](https://prnt.sc/viwbbp) 

  - potrebno je tekst zarotirati
  - metoda koja ce odrediti ugao pravca je **regresija** 

### Algoritam

  - nadjemo region ([bounderRectangle](https://prnt.sc/viwc6u) obican) 
  - odredimo [centre](https://prnt.sc/viwcu3) svake konture 
  - provucem [regresionu pravu](https://prnt.sc/viwd0q) kroz njih ( regresiona prava sadrzi informacije o uglu, tj globalnom uglu cele linije)
  - uzimam [ugao alfa](https://prnt.sc/viwdhh) ( arkus tanges od k ce nam dati ugao alfa) 
  - onda uzmemo celu sliku i uz pomoc opencv-a, rotiramo za minus alfa ( i on nam ispravi sliku )
  
</details>

<details>
  <summary> Fazi logika </summary> <br>

## Fazi logika 

Na ovom [primeru](https://prnt.sc/viwf6d) vidimo gde je regularni broj 5 zapravo 5, dok je u fazi logici on zapravo skup *oko petice*

### Gde je mi koristimo

Mi je koristimo u poredjenju stringova. Koristimo **Levenstajn** [algoritam](https://prnt.sc/viwjib) . U postprocesingu.

  - a mi cemo imati recnik svih reci
  - i onda cemo nase reci koje su falicne pokusavati da [namapiramo](https://prnt.sc/viwkbi) na reci koje su najslicnije u tom recniku 


### Fuzywazy

U njemu imamo opciju da indeksiramo recnik. I onda nam on automatski za neku datu rec vrati najslicniju rec.

</details>


<details>
  <summary> Klasterovanje slova i ostalih kontura </summary> <br>
  
  - [ovako](https://prnt.sc/vlun2i) nekako
  
</details>
