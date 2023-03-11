import pickle
import cfg
import utils
import prepare_model
from utils import normalize_and_reshape_inputs
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping


def train_network():
    sequence_length = cfg.sequence_length
    network_input, network_output = load_sequences(sequence_length)
    if cfg.model_remake:
        n_vocab = utils.get_n_vocab()
        model = prepare_model.create_model(sequence_length, n_vocab)
    else:
        model = load_model(filepath=cfg.best_weights)

    train(model, network_input, network_output)


def load_sequences(sequence_length):
    """METODA VRACA PAROVE ULAZ IZLAZ GDE JE ULAZ NIZ OD 100 NOTA NORMALIZOVAN, A IZLAZ ONE HOT CODED NIZ"""
    with open('data/train/input_train', 'rb') as filepath:
        input = pickle.load(filepath)
    with open('data/train/output_train', 'rb') as filepath:
        output = pickle.load(filepath)

    normalized_input = normalize_and_reshape_inputs(input,  sequence_length)

    return normalized_input, output


def train(model, network_input, network_output):
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
                                   min_delta=0.01,  # najmanja promena u odnosu na prethodnu epohu
                                   patience=3,
                                   # broj epoha sa promenom manjom od delta vrednosti nakon ceka trening staje
                                   verbose=1,  # da prikaze zasto je trening zaustavljen
                                   restore_best_weights=True)

    callbacks_list = [checkpoint, early_stopping]

    model.fit(network_input, network_output,
              epochs=250, initial_epoch=initial_epoch, validation_split=0.1,
              batch_size=cfg.batch_size,  # koliko ulaza se gura kroz mrezu
              callbacks=callbacks_list)


if __name__ == '__main__':
    train_network()
