import pickle
import cfg
import utils
import prepare_model
from utils import normalize_and_reshape_inputs
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping


def train_network():
    sequence_length = cfg.sequence_length
    n_vocab = utils.get_n_vocab()
    train_input, test_input, train_output, test_output = load_sequences(sequence_length)
    best_weights = cfg.best_weights
    #model = load_model(filepath="model/model_conf.hdf5")
    model = prepare_model.create_model(sequence_length, n_vocab)
    model.load_weights(best_weights) # JER JE TRENING MNOGO DUG VREMENSKI,
    # UCITAVA SE WEIGHTS SA NAJMANJOM GRESKOM DA SE NASTAVI
    train(model, train_input, test_input, train_output, test_output)


def load_sequences(sequence_length):
    """METODA VRACA PAROVE ULAZ IZLAZ GDE JE ULAZ NIZ OD 100 NOTA NORMALIZOVAN, A IZLAZ ONE HOT CODED NIZ"""
    with open('data/train/input', 'rb') as filepath:
        train_input = pickle.load(filepath)
    with open('data/test/input', 'rb') as filepath:
        test_input = pickle.load(filepath)
    with open('data/train/output', 'rb') as filepath:
        train_output = pickle.load(filepath)
    with open('data/test/output', 'rb') as filepath:
        test_output = pickle.load(filepath)

    normalized_train_input, normalized_test_input = normalize_and_reshape_inputs(train_input, test_input, sequence_length)

    return normalized_train_input, normalized_test_input, train_output, test_output


def train(model, network_input, val_input, network_output, val_output):
    filepath = cfg.weights_format
    initial_epoch = cfg.initial_epoch
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',  # loss je parametar treniranja
        verbose=1,  # 1 da prikazuje svaki callback, ne treba
        save_best_only=True,  # svaka iteracija pamti samo najbolje tezine
        mode='min'  # ako treba da se overwrituje fajl, zapamti sa manjim lossom
    )
    # early stopping je callback koji zaustavlja trening ako je mreza istrenirana pre isteka broja epoha treninga
    early_stopping = EarlyStopping(monitor='loss',
                                   min_delta=0.02,  # najmanja promena u odnosu na prethodnu epohu
                                   patience=3,
                                   # broj epoha sa promenom manjom od delta vrednosti nakon ceka trening staje
                                   verbose=1,  # da prikaze zasto je trening zaustavljen
                                   restore_best_weights=True)

    callbacks_list = [checkpoint, early_stopping]

    model.fit(network_input, network_output,
              validation_data=(val_input, val_output),  # podaci za validaciju nakon svake epohe
              epochs=200, initial_epoch=initial_epoch,
              batch_size=128  # koliko ulaza se gura kroz mrezu, ali izgleda da je za tensorflow cpu nebitno
              , callbacks=callbacks_list)


if __name__ == '__main__':
    train_network()
