# GENERATOR MUZIKE
## Uvod

Neuronska mreža koja stvara od sekvence ulaza (nota ili akorda) predviđa izlaz (notu ili akord). Krajnji izlaz je MIDI fajl koji može da se reprodukuje.

Ulazni podaci su kompozicije iz video igrica koje imaju sličan tempo i tonalitet.

## Trening
Za pripremu podataka za trening pokreće se skripta 'prepare_data.py' koja deserijalizuje MIDI fajlove
, kreira parove ulaz izlaz, deli takve podatke na deo za trening i deo za test, onda ih ponovo serijalizuje u binarne fajlove. Ovo se radi samo jednom za svaki unikatni model.

Za postupak treniranja mreže pokreće se skripta 'lstm.py'.

Ova skripta će prvo da deserijalizuje sve fajlove iz 'data/test' i 'data/train' i da normalizuje ulazne podatke.

Nakon toga  stvara model i kompajlira i na kraju pokreće trening.

Model mreže se prikazuje u 'model/model.png' (pomoću keras metode plot_model), težine za svaku epohu se nalaze u 'weights' direktorijumu (na githubu samo krajnja jer su fajlovi ogromni)



## Stvaranje kompozicije

Kada se mreža istrenira pokreće se skripta 'predict.py'


U metodi model.load_weights kao argument se stavlja putanja do težina sa najmanjom greškom, u ovom slučaju: **"weights/weights-improvement-10-0.1891-bigger.hdf5"** 

Metoda generate_notes predviđa n sledećih nota ili akorda od početne sekvence koja je nasumično izabrana iz test dataset-a.