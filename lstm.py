""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.initializers import RandomUniform
from keras.layers.normalization import BatchNormalization


#RANDOM_SEED = 42

#numpy.random.seed(seed=RANDOM_SEED)


def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)

        #print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    print("Parsing done")
    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    #network_input = network_input / float(n_vocab)

    #network_input = network_input / numpy.linalg.norm(network_input)

    network_input = (network_input - network_input.mean()) / network_input.std()
    
    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """

    initializer = RandomUniform(minval=0.001, maxval=0.05)

# neural network with 5 layers
#
# · · · · · · · · · ·          (input data, flattened pixels)       X [batch, 784]   # 784 = 28*28
# \x/x\x/x\x/x\x/x\x/       -- fully connected layer (sigmoid+BN)   W1 [784, 200]      B1[200]
#  · · · · · · · · ·                                                Y1 [batch, 200]
#   \x/x\x/x\x/x\x/         -- fully connected layer (sigmoid+BN)   W2 [200, 100]      B2[100]
#    · · · · · · ·                                                  Y2 [batch, 100]
#     \x/x\x/x\x/           -- fully connected layer (sigmoid+BN)   W3 [100, 60]       B3[60]
#      · · · · ·                                                    Y3 [batch, 60]
#       \x/x\x/             -- fully connected layer (sigmoid+BN)   W4 [60, 30]        B4[30]
#        · · ·                                                      Y4 [batch, 30]
#         \x/               -- fully connected layer (softmax+BN)   W5 [30, 10]        B5[10]
#          ·                                                        Y5 [batch, 10]


    model = Sequential()
    
    ### Layer 1
    model.add(LSTM(
        int(network_input.shape[1]),
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    ### Layer 2
    model.add(LSTM(int(n_vocab*1.3**2), return_sequences=True, dropout=0.3, recurrent_dropout=0.3))
    model.add(BatchNormalization())
    #model.add(Dropout(0.3))

    ### Layer 3
    model.add(LSTM(int(n_vocab*1.3**2), dropout=0.3, recurrent_dropout=0.3))
    model.add(BatchNormalization())

    ### Layer 4
    model.add(Dense(int(n_vocab*1.3), kernel_initializer='lecun_normal', bias_initializer=initializer))
    model.add(BatchNormalization())
    model.add(Activation('selu'))
    model.add(Dropout(0.3))

    ### Layer 5
    model.add(Dense(n_vocab, kernel_initializer='lecun_normal', bias_initializer=initializer))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.load_weights('weights-improvement-49-0.8612-bigger.hdf5')

    return model

def train(model, network_input, network_output):

    BATCH_SIZE = 100


    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=BATCH_SIZE, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)
    callbacks_list = [checkpoint] #[checkpoint, tensorboard]

    model.fit(network_input, network_output, epochs=200, batch_size=BATCH_SIZE, callbacks=callbacks_list, validation_split=0.3,
        initial_epoch=49)


""" Train a Neural Network to generate music """
notes = get_notes()

# get amount of pitch names
n_vocab = len(set(notes))

network_input, network_output = prepare_sequences(notes, n_vocab)

model = create_network(network_input, n_vocab)

train(model, network_input, network_output)