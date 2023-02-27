import pickle
import cfg
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping


def train_network():
    # get_notes()
    # notes je binary fajl gde su serijalizovane note koje su parsirane prethodno
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)  # pickle.load deserijalizuje taj fajl u numpy array

    # ovo je broj unikatnih nota, parametar za treniranje
    n_vocab = len(set(notes))
    sequence_length = cfg.sequence_length
    train_input, test_input, train_output, test_output = load_sequences(n_vocab, sequence_length)

    model = create_model(sequence_length, n_vocab)
    # model.load_weights("weights/weights-improvement-10-0.1891-bigger.hdf5") # JER JE TRENING MNOGO DUG VREMENSKI,
    # UCITAVA SE WEIGHTS SA NAJMANJOM GRESKOM DA SE NASTAVI
    train(model, train_input, test_input, train_output, test_output)


def normalize_and_reshape_inputs(train_input, test_input, sequence_length, n_vocab):
    length_train = len(train_input)  # ukupan broj nota za treniranje (oko 36000)
    length_test = len(test_input)  # ukupan broj nota za validaciju (oko 9000)

    normalized_train_input = numpy.reshape(train_input, (length_train, sequence_length, 1)) # niz se pretvara u niz od 100 nizova
    normalized_test_input = numpy.reshape(test_input, (length_test, sequence_length, 1))

    normalized_test_input = normalized_test_input / float(n_vocab) # svaki ulaz ima int value od 0 do 326, normalizuje se tako sto se svaka int vrednost deli sa max 
    normalized_train_input = normalized_train_input / float(n_vocab)

    return normalized_train_input, normalized_test_input


def load_sequences(n_vocab, sequence_length):
    """METODA VRACA PAROVE ULAZ IZLAZ GDE JE ULAZ NIZ OD 100 NOTA NORMALIZOVAN, A IZLAZ ONE HOT CODED NIZ"""
    with open('data/train/input', 'rb') as filepath:
        train_input = pickle.load(filepath)
    with open('data/test/input', 'rb') as filepath:
        test_input = pickle.load(filepath)
    with open('data/train/output', 'rb') as filepath:
        train_output = pickle.load(filepath)
    with open('data/test/output', 'rb') as filepath:
        test_output = pickle.load(filepath)

    normalized_train_input, normalized_test_input = normalize_and_reshape_inputs(train_input, test_input,
                                                                                 sequence_length, n_vocab)

    return normalized_train_input, normalized_test_input, train_output, test_output


def create_model(sequence_length, n_vocab):
    """ slojevi: LSTM ulazni, LSTM, LSTM, Dropout, Dense, Dropout, Dense izlazni,  funkcija gubitka categorical
    crossentropy,  metrika za validaciju: CATEGORICAL ACCURACY, optimizator: ADAM SA PARAMETRIMA LEARNING RATE 0.001 """
    dropout_seed = cfg.dropout_seed
    loss_function = CategoricalCrossentropy()
    metrics = [CategoricalAccuracy()]
    optimizer = Adam(learning_rate=0.001)
    model = Sequential(name="neural composer")
    model.add(LSTM(
        512,
        name="input LSTM",
        input_shape=(sequence_length, 1),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, name="LSTM2", return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512, name="LSTM3"))
    model.add(BatchNorm())
    model.add(Dropout(rate=0.3, seed=dropout_seed))  # dropout layer setuje 30% random ulaza na nula  tako smanjuje broj ulaza, a one koje nije
    # setovao na 0 mnozi sa 1-0.3 da bi ukupan ulaz ostao isti, sprecava overfitting
    model.add(Dense(256, name="DENSE256"))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(rate=0.3, seed=dropout_seed))
    model.add(Dense(n_vocab, name="output DENSE"))
    model.add(Activation('softmax'))
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)

    return model


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
                                   min_delta=0.02, # najmanja promena u odnosu na prethodnu epohu
                                   patience=3,  # broj epoha sa promenom manjom od delta vrednosti nakon ceka trening staje
                                   verbose=1,  # da prikaze zasto je trening zaustavljen
                                   restore_best_weights=True)
    
    callbacks_list = [checkpoint, early_stopping]

    model.fit(network_input, network_output, 
              validation_data=(val_input, val_output), # podaci za validaciju nakon svake epohe
              epochs=200, initial_epoch=initial_epoch,
              batch_size=128  # koliko ulaza se gura kroz mrezu, ali izgleda da je za tensorflow cpu nebitno
              , callbacks=callbacks_list)


if __name__ == '__main__':
    train_network()
