from numpy import reshape
import pickle

import cfg


def normalize_and_reshape_inputs(input, sequence_length):
    """NORMALIZUJE SVE ULAZE"""
    length_input = len(input)  # ukupan broj nota za

    normalized_input = reshape(input, (length_input, sequence_length, 1))  # niz se pretvara u niz od 100 nizova
    max_value = get_max_value()
    # se tako sto se svaka int vrednost deli sa max
    normalized_input = normalized_input / float(max_value)

    return normalized_input


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
    max_value = get_n_vocab()-1
    return max_value


def get_pitchnames():
    """OVA METODA VRACA TUPLE UNIKATNIH NOTA U CELOM DATASETU"""
    with open('data/pitchnames', 'rb') as filepath:
        pitchnames = pickle.load(filepath)
    return pitchnames

def get_test_data():
    with open('data/test/input_test', 'rb') as filepath:
        input = pickle.load(filepath)
    with open('data/test/output_test', 'rb') as filepath:
        output = pickle.load(filepath)
    return input, output

