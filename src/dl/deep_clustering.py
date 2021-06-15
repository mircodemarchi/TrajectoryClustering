import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd

from util.util import get_elapsed
from util.log import Log


class LSTMEncoder(tf.keras.Model):
    def __init__(self, enc_units, dense_units=None):
        super(LSTMEncoder, self).__init__()
        self.dense_units = dense_units
        self.enc_units = enc_units
        self.dense = tf.keras.layers.Dense(dense_units) if dense_units else None
        self.lstm = tf.keras.layers.LSTM(self.enc_units, return_state=True, return_sequences=True)
        self.state = None

    def call(self, x, hidden=None, training=None, mask=None):
        if self.dense:
            x = self.dense(x)
        output, state_m, state_c = self.lstm(x, initial_state=hidden)
        self.state = (state_m, state_c)
        return output

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

    def get_state(self):
        return self.state

    def get_config(self):
        return {"enc_units": self.enc_units, "dense_units": self.dense_units}


class LSTMDecoder(tf.keras.Model):
    def __init__(self, dec_units, output_units=None):
        super(LSTMDecoder, self).__init__()
        self.dec_units = dec_units
        self.output_units = output_units
        self.lstm = tf.keras.layers.LSTM(self.dec_units, return_state=True, return_sequences=True)
        self.dense_output = tf.keras.layers.Dense(self.output_units) if output_units else None

    def call(self, x, hidden=None, training=None, mask=None):
        output, _ = self.lstm(x, initial_state=hidden)
        if self.dense_output:
            output = self.dense_output(output)
        return output

    def get_config(self):
        return {"dec_units": self.dec_units, "output_units": self.output_units}


class LSTMAddonsDecoder(tf.keras.Model):
    def __init__(self, dec_units, output_units=None):
        super(LSTMAddonsDecoder, self).__init__()
        self.dec_units = dec_units
        self.output_units = output_units
        self.dense_output = tf.keras.layers.Dense(self.output_units) if output_units else None

        self.sampler = tfa.seq2seq.TrainingSampler()
        self.decoder_cell = tf.keras.layers.LSTMCell(dec_units)
        self.decoder = tfa.seq2seq.BasicDecoder(self.decoder_cell, self.sampler, self.dense_output)

    def call(self, x, hidden=None, training=None, mask=None):
        output, _, _ = self.decoder(x, initial_state=hidden)
        return output.rnn_output

    def get_config(self):
        return {"dec_units": self.dec_units, "output_units": self.output_units}


class LSTMAutoregressiveDecoder(tf.keras.Model):
    def __init__(self, dec_units, output_steps, output_units=None):
        super(LSTMAutoregressiveDecoder, self).__init__()
        self.dec_units = dec_units
        self.output_steps = output_steps
        self.output_units = output_units

        self.lstm_cell = tf.keras.layers.LSTMCell(dec_units)
        self.lstm = tf.keras.layers.RNN(self.lstm_cell, return_state=True)

        self.dense_output = tf.keras.layers.Dense(self.output_units) if output_units else None

    def warmup(self, inputs, state):
        # inputs.shape => (batch, time, features)
        # prediction.shape => (batch, lstm_units)
        prediction, *state = self.lstm(inputs, initial_state=state)

        # prediction.shape => (batch, features)
        if self.dense_output:
            prediction = self.dense_output(prediction)
        return prediction, state

    def call(self, x, hidden=None, training=None, mask=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the lstm state
        x, state = self.warmup(x, hidden)
        # Insert the first prediction
        predictions.append(x)

        # Run the rest of the prediction steps
        for n in range(1, self.output_steps):
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state, training=training)
            # Convert the lstm output to a prediction.
            if self.dense_output:
                x = self.dense_output(x)
            # Add the prediction to the output
            predictions.append(x)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions

    def get_config(self):
        return {"dec_units": self.dec_units, "output_units": self.output_units}


class AutoEncoder(tf.keras.Model):
    def __init__(self, config, feature_dim, latent_dim=1):
        super(AutoEncoder, self).__init__()
        self.config = config
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

    def call(self, x, hidden=None, training=None, mask=None):
        pass


class DeepClustering:
    def __init__(self, data: pd.DataFrame):
        self.data = tf.data.Dataset.from_tensor_slices((data, data))
