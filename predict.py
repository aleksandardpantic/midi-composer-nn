""" This module generates notes for a midi file using the
    trained neural network """
import pickle
import numpy
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Activation
from keras.utils import plot_model


def generate():
    # deserijalizuju se note
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # isto kao u lstm.py
    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(set(notes))

    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    model = create_network(normalized_input, n_vocab)  # ovde se stvara isti model kao u lstm.py
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)  # serijalizuje output u midi fajl


def prepare_sequences(notes, pitchnames, n_vocab):
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    normalized_input = normalized_input / float(n_vocab)

    return network_input, normalized_input


def create_network(network_input, n_vocab):
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
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # ubacuju se tezine iz  epohe  sa najmanjim lossom hdf5
    model.load_weights('weights/weights-improvement-32-0.2055-bigger.hdf5')
    # plot_model(model, "model/model.png", show_shapes=True)
    return model


def generate_notes(model, network_input, pitchnames, n_vocab):
    # random sekvenca ulaza kao pocetna tacka za predikciju
    start = numpy.random.randint(0, len(network_input) - 1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # model generise 500 nota, neke su i akordi
    for note_index in range(100):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=1)  # verbose bi trebalo da ispisuje detalje u konzoli?

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


def create_midi(prediction_output):
    offset = 0
    output_notes = []

    for pattern in prediction_output:

        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.ChurchBells()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.ChurchBells()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='output/ChurchBells.mid')


if __name__ == '__main__':
    generate()
