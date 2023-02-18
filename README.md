# GENERATOR MUZIKE

Neuronska mreža koja stvara od sekvence ulaza (nota) predviđa izlaz (notu). Krajnji izlaz je MIDI fajl koji može da se reprodukuje.

Ulazni podaci su pesme iz nekih video igrica koje imaju sličan tempo i tonalitet.

## Trening

Za postupak treniranja mreže pokreće se skripta lstm.py

Ova skripta će prvo serijalizovati sve fajlove iz '/midi_songs' u 'data/notes', da bi svaki sledeći trening sa istim podacima bio brži.

Nakon toga stvara sekvence ulaza i izlaza, stvara model i kompajlira i na kraju pokreće trening.

Model mreže se prikazuje u 'model/model.png' (pomoću keras metode plot_model), težine za svaku epohu se nalaze u 'weights' direktorijumu (na githubu samo krajnja jer su fajlovi ogromni)



## Stvaranje kompozicije

Kada se mreža istrenira pokreće se skripta 'predict.py'


U metodi model.load_weights kao argument se stavlja putanja do težina sa najmanjom greškom, u ovom slučaju: **"weights/weights-improvement-10-0.1891-bigger.hdf5"** 
