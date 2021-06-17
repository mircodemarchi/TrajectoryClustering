import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd

from util.util import get_elapsed
from util.log import Log


class LSTMEncoder(tf.keras.Model):
    def __init__(self, enc_units, batch_sz=1, dense_units=None):
        super(LSTMEncoder, self).__init__()
        self.batch_sz = batch_sz
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
        return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]

    def get_state(self):
        return self.state

    def get_config(self):
        return {"enc_units": self.enc_units, "batch_sz": self.batch_sz, "dense_units": self.dense_units}


class LSTMDecoder(tf.keras.Model):
    def __init__(self, dec_units, output_units=None):
        super(LSTMDecoder, self).__init__()
        self.dec_units = dec_units
        self.output_units = output_units

        self.lstm = tf.keras.layers.LSTM(self.dec_units, return_state=True, return_sequences=True)
        self.dense_output = tf.keras.layers.Dense(self.output_units) if output_units else None

    def call(self, x, hidden=None, training=None, mask=None):
        output, _, _ = self.lstm(x, initial_state=hidden)
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

        # Decoder
        self.sampler = tfa.seq2seq.TrainingSampler()
        self.decoder_cell = tf.keras.layers.LSTMCell(dec_units)
        self.decoder = tfa.seq2seq.BasicDecoder(self.decoder_cell, self.sampler, self.dense_output)

        # Dense output
        self.dense_output = tf.keras.layers.Dense(self.output_units) if output_units else None

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

        # Autoregressive module
        self.lstm_cell = tf.keras.layers.LSTMCell(dec_units)
        self.lstm = tf.keras.layers.RNN(self.lstm_cell, return_state=True)

        # Dense output
        self.dense_output = tf.keras.layers.Dense(self.output_units) if output_units else None

    def warmup(self, inputs, state):
        # input.shape => (batch, features)
        # prediction.shape => (batch, lstm_units)
        prediction, state = self.lstm_cell(inputs, states=state)

        if self.dense_output:
            # prediction.shape => (batch, features)
            prediction = self.dense_output(prediction)
        return prediction, state

    def call(self, last_input, hidden=None, training=None, mask=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the lstm state
        x, state = self.warmup(last_input, hidden)
        # Insert the first prediction
        predictions.append(x)

        # Run the rest of the prediction steps
        for n in range(1, self.output_steps):
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state, training=training)

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
        return {"dec_units": self.dec_units, "output_steps": self.output_steps, "output_units": self.output_units}


class AutoEncoder(tf.keras.Model):
    """
    config: "autoregressive", "addons", "simple"
    """
    def __init__(self, time_dim, feature_dim, config=None, latent_dim=1, hidden_dim=None):
        super(AutoEncoder, self).__init__()
        self.time_dim = time_dim
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.config = config if config else "autoregressive"

        self.encoder = LSTMEncoder(enc_units=latent_dim, dense_units=hidden_dim)
        if self.config == "simple":
            self.decoder = LSTMDecoder(dec_units=latent_dim if hidden_dim else feature_dim,
                                       output_units=feature_dim if hidden_dim else None)
        elif self.config == "addons":
            self.decoder = LSTMAddonsDecoder(dec_units=latent_dim if hidden_dim else feature_dim,
                                             output_units=feature_dim if hidden_dim else None)
        else:
            # config == "autoregressive"
            self.decoder = LSTMAutoregressiveDecoder(dec_units=latent_dim if hidden_dim else feature_dim,
                                                     output_steps=time_dim,
                                                     output_units=feature_dim if hidden_dim else None)

    def call(self, x, hidden=None, training=None, mask=None):
        init_state = self.encoder.initialize_hidden_state()
        encoded = self.encoder(x, hidden=init_state)
        state = self.encoder.get_state()
        if self.config == "autoregressive":
            # Insert the last input as decoder input.
            decoded = self.decoder(x[:, -1, :], hidden=state)
        else:
            decoded = self.decoder(encoded, hidden=state)
        return decoded

    def get_config(self):
        return {"time_dim": self.time_dim, "feature_dim": self.feature_dim, "config": self.config,
                "latent_dim": self.latent_dim, "hidden_dim": self.hidden_dim}

    def get_latent_state(self):
        return self.encoder.get_state()


class DeepClustering:
    def __init__(self, data: pd.DataFrame, model: AutoEncoder, seq_len, epoch=10, batch_sz=1):
        self.seq_len = seq_len
        self.epoch = epoch
        self.batch_sz = batch_sz
        self.data = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data.to_numpy(),
            targets=None,
            sequence_length=seq_len,
            sequence_stride=1,
            shuffle=True,
            batch_size=batch_sz)
        self.model = model

    def train(self, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')

        self.model.compile(loss=tf.losses.MeanSquaredError(),
                           optimizer=tf.optimizers.Adam(),
                           metrics=[tf.metrics.MeanAbsoluteError()])

        history = self.model.fit(self.data, self.data, epochs=self.epoch,
                                 callbacks=[early_stopping])
        return history

    def get_train_state(self):
        train_loss, train_acc = self.model.evaluate(self.data, self.data)
        return train_loss, train_acc

    def get_latent_state(self):
        return self.model.get_latent_state()
