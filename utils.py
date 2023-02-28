from numpy import reshape
import pickle


def normalize_and_reshape_inputs(train_input, test_input, sequence_length):
    """NORMALIZUJE SVE ULAZE"""
    length_train = len(train_input)  # ukupan broj nota za treniranje (oko 36000)
    length_test = len(test_input)  # ukupan broj nota za validaciju (oko 9000)

    normalized_train_input = reshape(train_input,
                                     (length_train, sequence_length, 1))  # niz se pretvara u niz od 100 nizova
    normalized_test_input = reshape(test_input, (length_test, sequence_length, 1))
    max_value = get_max_value()
    normalized_test_input = normalized_test_input / float(max_value)  # svaki ulaz ima int value od 0 do n, normalizuje
    # se tako sto se svaka int vrednost deli sa max
    normalized_train_input = normalized_train_input / float(max_value)

    return normalized_train_input, normalized_test_input


def note_to_int():
    """SVAKOM MOGUCEM UNIKATNOM IZLAZU SE DODELJUJE INT VREDNOST, OVA METODA VRACA DICTIONARY NPR {'A1': 0, 'B3': 1,
    'C#4': 2, '0.1.5': 3}"""
    with open('data/pitchnames', 'rb') as filepath:
        pitchnames = pickle.load(filepath)
    note_to_int_dict = dict((note, number) for number, note in enumerate(pitchnames))
    return note_to_int_dict


def int_to_note():
    """PREVODI INT VREDNOST SVAKE NOTE U SIMBOLICKU, VRACA DICTIONARY NPR {0: 'A1', 1: 'B3', 2: 'C#4', 3: '0.1.5'}"""
    with open('data/pitchnames', 'rb') as filepath:
        pitchnames = pickle.load(filepath)

    int_to_note_dict = dict(
        (number, note) for number, note in enumerate(pitchnames))

    return int_to_note_dict


def get_n_vocab():
    """OVA METODA VRACA INT BROJ MOGUCIH IZLAZNIH NOTA"""
    with open('data/pitchnames', 'rb') as filepath:
        pitchnames = pickle.load(filepath)

    n_vocab = len(set(pitchnames))
    return n_vocab


def get_max_value():
    """VRACA INT BROJ MAKSIMALNE INT VREDNOSTI NOTE, ZA NORMALIZACIJU ULAZA"""
    max_value = max(int_to_note().keys())
    return max_value


def get_pitchnames():
    """OVA METODA VRACA TUPLE UNIKATNIH NOTA U CELOM DATASETU"""
    with open('data/pitchnames', 'rb') as filepath:
        pitchnames = pickle.load(filepath)
    return pitchnames

