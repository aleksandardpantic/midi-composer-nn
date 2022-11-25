""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


def train_network():
    # notes je binary fajl gde su serijalizovane note koje su parsirane prethodno
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)  # pickle.load deserijalizuje taj fajl u numpy array

    # ovo je broj unikatnih nota, parametar za treniranje
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)
    model.load_weights("weights/weights-improvement-32-0.2055-bigger.hdf5")
    train(model, network_input, network_output)


def get_notes():  # ovo se radi samo jednom, posle se cita iz fajla

    notes = []
    # podaci su odvojeni u dve klase: Note i Chord. Note objekat sadrzi podatke o tri parametra, pitch,  octave,
    # and offset. Chord je container za 3 ili vise Notes objekata
    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try:  # ako fajl ima instrumente, jer su neki corrupted
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # neki fajlovi nemaju instrumente samo note???
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))  #
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))  # akord pretvara u note odvojene tackom

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab):
    # broj sekvenci, ne valja da bude mali zbog exploding gradijent i greske su velike, lstm pamti 100 nota unazad,
    # vise od 100 je pretesko za treniranje, treba mnogo vremena
    sequence_length = 100

    # poredjani pojedinacni pitch
    pitchnames = sorted(set(item for item in notes))

    # dictionary koji prevodi pitch u int
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # stvara sekvence
    # za predvidjanje jedne note ili akorda, koristi 100 prethodnih, povezuju se parovi 100 ulaza jedan izlaz
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # da bi bio kompatibilan sa lstm sojevima
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalizuje input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)  # pretvara niz u binarnu matricu 0 i 1, za categorical
    # crossentropy je neophodno

    return network_input, network_output


def create_network(network_input, n_vocab):
    """ slojevi: LSTM ulazni, LSTM, LSTM, Dropout, Dense, Dropout, Dense izlazni,  funkcija gubitka categorical
    crossentropy, ne znam koji je ovo optimizator: RMSPROP """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3, ))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))  # dropout layer setuje random ulaze na nula po rejtu 1-1/0.3 tako smanjuje broj ulaza,
    # sprecava overfitting
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def train(model, network_input, network_output):
    filepath = "weights/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',  # loss je parametar treniranja
        verbose=1,  # 1 da prikazuje svaki callback, ne treba
        save_best_only=True,  # svaka iteracija pamti samo najbolje tezine
        mode='min'  # ako treba da se overwrituje fajl, zapamti sa manjim lossom
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=200, batch_size=128  # koliko ulaza se gura kroz mrezu, ali
              # izgleda da je za tensorflow cpu nebitno
              , callbacks=callbacks_list)


if __name__ == '__main__':
    train_network()
