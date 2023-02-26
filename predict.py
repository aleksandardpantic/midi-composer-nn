import pickle
import numpy
import cfg
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Activation


def generate():
    # deserijalizuju se note
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # isto kao u lstm.py
    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(set(notes))
    sequence_length = 100
    with open('data/test/input', 'rb') as filepath:
        network_input = pickle.load(filepath)
    model = create_network(sequence_length, n_vocab)  # ovde se stvara isti model kao u lstm.py
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)  # serijalizuje output u midi fajl


def create_network(sequence_length, n_vocab):
    best_weights = cfg.best_weights
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(sequence_length, 1),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3, ))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # ubacuju se tezine iz  epohe  sa najmanjim lossom hdf5
    model.load_weights(best_weights)  # OVDE SE OVEK STAVLJAJU TEŽINE SA NAJMANJOM GREŠKOM
    return model


def generate_notes(model, network_input, pitchnames, n_vocab):
    # random sekvenca ulaza kao pocetna tacka za predikciju
    start = numpy.random.randint(0, len(network_input) - 1)  # random broj koji odredjuje pocectnu sekvencu od 100 na
    # osnovu koje se predvidja prva nota

    int_to_note = dict(
        (number, note) for number, note in enumerate(pitchnames))  # dictionary koji prevodi int u ime note

    pattern = network_input[start]  # pocetna sekvenca na osnovu koje se predvidja
    prediction_output = []

    # model generise 500 nota, neke su i akordi
    for note_index in range(30):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=1)  # verbose bi trebalo da ispisuje detalje u konzoli?

        index = numpy.argmax(prediction)  # vrednost aktiviranog neurona je priblizna 1, ovde se uzima index predvidjene note
        result = int_to_note[index]  # int vrednost  se prebacuje u string, npr 'C7'
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(
            pattern)]  # pomera se za 1 u desno, i vec generisana nota se uzima u obzir pri sledecem predvidjanju

    return prediction_output


def create_midi(prediction_output):
    offset = 0
    output_notes = []
    output_file = cfg.output_stream_file
    for pattern in prediction_output:

        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Banjo()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Banjo()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)


if __name__ == '__main__':
    generate()
