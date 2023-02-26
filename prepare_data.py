import pickle
from keras.utils import np_utils
from sklearn import model_selection


def prepare_sequences():
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)
    # broj sekvenci, ne valja da bude mali zbog exploding gradijent i greske su velike, lstm pamti 100 nota unazad,
    # vise od 100 je pretesko za treniranje, treba mnogo vremena
    sequence_length = 100
    n_vocab = len(set(notes))
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

    network_output = np_utils.to_categorical(network_output)  # pretvara niz u binarnu matricu 0 i 1, za categorical
    # crossentropy je neophodno
    input_train, input_test, output_train, output_test = model_selection.train_test_split(network_input, network_output,
                                                                                          test_size=0.2,
                                                                                          random_state=134)

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
    prepare_sequences()
