import cfg
from keras.models import Sequential, save_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.optimizers import Adam
from keras.utils import plot_model


def create_model(sequence_length, n_vocab):
    """ slojevi: LSTM ulazni, LSTM, LSTM, Dropout, Dense, Dropout, Dense, Dropout, Dense izlazni,  funkcija gubitka categorical
    crossentropy,  metrika za validaciju: CATEGORICAL ACCURACY, optimizator: ADAM SA PARAMETRIMA LEARNING RATE 0.001 """
    dropout_seed = cfg.dropout_seed
    loss_function = CategoricalCrossentropy()
    metrics = [CategoricalAccuracy()]
    optimizer = Adam(learning_rate=0.001)
    model = Sequential(name="midi_composer_nn")
    model.add(LSTM(
        512,
        name="LSTM1",
        input_shape=(sequence_length, 1),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, name="LSTM2", return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512, name="LSTM3"))
    model.add(BatchNorm())
    model.add(Dropout(rate=0.3, seed=dropout_seed, name="dropout1"))  # dropout layer setuje 30% random ulaza na nula
    #  tako smanjuje broj ulaza, a one koje nije setovao na 0 mnozi sa 1-0.3 da bi ukupan ulaz ostao isti, sprecava overfitting
    model.add(Dense(512, name="DENSE1"))
    model.add(Activation('relu', name="relu"))
    model.add(BatchNorm())
    model.add(Dropout(rate=0.3, seed=dropout_seed, name="dropout2"))
    model.add(Dense(256, name="DENSE2"))
    model.add(Activation('relu', name="relu2"))
    model.add(BatchNorm())
    model.add(Dropout(rate=0.3, seed=dropout_seed, name="dropout3"))
    model.add(Dense(n_vocab, name="output_DENSE"))
    model.add(Activation('softmax', name="softmax"))
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    plot_model(model, to_file="model/model.png", show_shapes=True)
    save_model(model=model, filepath="model/model_conf.hdf5", overwrite=True)
    return model
