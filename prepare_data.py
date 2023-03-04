import pickle
import glob
from keras.utils import np_utils
from sklearn import model_selection
from music21 import instrument, note, chord, converter
import cfg


def get_notes():
    """OVA METODA PARSIRA SVE MIDI FAJLOVE U NIZ I SERIJALIZUJE IH U FAJL"""

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

    pitchnames = sorted(set(item for item in notes))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    with open('data/pitchnames', 'wb') as filepath:
        pickle.dump(pitchnames, filepath)


def prepare_sequences():
    """METODA SERIJALIZUJE PAROVE ULAZ IZLAZ GDE JE ULAZ NIZ OD 100 NOTA, A IZLAZ ONE HOT CODED NIZ, ZA TRAIN I TEST DEO"""
    sequence_length = cfg.sequence_length
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)
    # broj sekvenci, ne valja da bude mali zbog exploding gradijent i greske su velike, lstm pamti 100 nota unazad,
    # vise od 100 je pretesko za treniranje, treba mnogo vremena

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

    # da bi bio kompatibilan sa lstm sojevima
    network_output = np_utils.to_categorical(network_output)  # pretvara niz u binarnu matricu 0 i 1, za categorical
    # crossentropy je neophodno
    input_train, input_test, output_train, output_test = model_selection.train_test_split(network_input, network_output,
                                                                                          test_size=0.1,
                                                                                          random_state=100)

    with open('data/train/input', 'wb') as filepath:
        pickle.dump(input_train, filepath)
    with open('data/train/output', 'wb') as filepath:
        pickle.dump(output_train, filepath)

    with open('data/test/input', 'wb') as filepath:
        pickle.dump(input_test, filepath)

    with open('data/test/output', 'wb') as filepath:
        pickle.dump(output_test, filepath)
    return


if __name__ == '__main__':
    get_notes()
    prepare_sequences()
