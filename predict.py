import pickle
import numpy
import cfg
from music21 import instrument, note, stream, chord
from keras.models import Model, load_model

import utils


def generate():
    max_value = utils.get_max_value()
    with open('data/test/input', 'rb') as filepath:
        network_input = pickle.load(filepath)
    model = load_model(filepath="model/model_conf.hdf5")
    model.load_weights(cfg.best_weights)
    prediction_output = generate_notes(model, network_input, max_value)
    create_midi(prediction_output)  # serijalizuje output u midi fajl


def generate_notes(model, network_input, max_value):
    # random sekvenca ulaza kao pocetna tacka za predikciju
    start = numpy.random.randint(0, len(network_input) - 1)  # random broj koji odredjuje pocectnu sekvencu od 100 na
    # osnovu koje se predvidja prva nota

    int_to_note = utils.int_to_note()

    pattern = network_input[start]  # pocetna sekvenca na osnovu koje se predvidja
    prediction_output = []
    n_predictions = cfg.number_of_predictions
    # model generise n_predictions nota, neke su i akordi
    for note_index in range(n_predictions):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(max_value)

        prediction = model.predict(prediction_input, verbose=1)  # verbose bi trebalo da ispisuje detalje u konzoli?

        index = numpy.argmax(prediction)  # vrednost aktiviranog neurona je priblizna 1, ovde se uzima index predvidjene note
        result = int_to_note[index]  # int vrednost  se prebacuje u string, npr 'C7'
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]  # pomera se za 1 u desno, i vec generisana nota se uzima u obzir pri sledecem predvidjanju

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
