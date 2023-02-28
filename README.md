# GENERATOR MUZIKE
## Uvod

Neuronska mreža koja stvara od sekvence ulaza (nota ili akorda) predviđa izlaz (notu ili akord). Krajnji izlaz je MIDI fajl koji može da se reprodukuje.

Ulazni podaci su kompozicije iz video igrica koje imaju sličan tempo i tonalitet.

## Priprema modela i podataka

Za pripremu podataka za trening pokreće se skripta 'prepare_data.py' koja deserijalizuje MIDI fajlove
, kreira parove ulaz izlaz, deli takve podatke na deo za trening i deo za test, onda ih ponovo serijalizuje u binarne fajlove u odgovarajućim direktorijumima. Ovo se radi samo jednom za svaki unikatni model.

Model mreže se prikazuje u 'model/model.png' (pomoću keras metode plot_model).

Za pripremu modela mreže se pokreće skripta 'prepare_model.py', koja kreira model, generiše png i serijalizuje ga u fajl 'model_conf.hdf5'

**Trening je vremenski dug i mora da se vrši u više navrata. Ovaj deo se radi samo jednom za svaku instancu treninga, tj. ako se menjaju model i/ili ulazni podaci, ovaj deo se izvršava ponovo.**

## Trening

Za postupak treniranja mreže pokreće se skripta 'lstm.py'.

Ova skripta će prvo da deserijalizuje sve fajlove iz 'data/test' i 'data/train' i da normalizuje ulazne podatke.

Učitava konfiguraciju modela, učitava težine ako je potrebno, i na kraju pokreće trening.

Težine za svaku epohu se nalaze u 'weights' direktorijumu (na githubu samo krajnja jer su fajlovi ogromni)



## Stvaranje kompozicije

Kada se mreža istrenira pokreće se skripta 'predict.py'


U metodi model.load_weights kao argument se stavlja putanja do težina sa najmanjom greškom, u ovom slučaju: **"weights/weights-improvement-10-0.1891-bigger.hdf5"** 

Metoda generate_notes predviđa n sledećih nota ili akorda od početne sekvence koja je nasumično izabrana iz test dataset-a.