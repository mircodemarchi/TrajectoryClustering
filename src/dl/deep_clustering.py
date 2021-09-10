import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np

from util.util import get_elapsed
from util.log import Log

log = Log(__name__, enable_console=True, enable_file=False)


class LSTMEncoder(tf.keras.layers.Layer):
    def __init__(self, enc_units, batch_sz=1, name="lstm_encoder", **kwargs):
        super(LSTMEncoder, self).__init__(name=name, **kwargs)
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.lstm = tf.keras.layers.LSTM(self.enc_units, return_state=True, return_sequences=True)

    def call(self, x, hidden=None, training=None):
        output, state_m, state_c = self.lstm(x, initial_state=hidden, training=training)
        return output, state_m, state_c

    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]

    def get_config(self):
        config = super(LSTMEncoder, self).get_config()
        config.update({"enc_units": self.enc_units, "batch_sz": self.batch_sz})
        return config


class LSTMDecoder(tf.keras.layers.Layer):
    def __init__(self, dec_units, output_units, name="lstm_decoder", **kwargs):
        super(LSTMDecoder, self).__init__(name=name, **kwargs)
        self.dec_units = dec_units
        self.output_units = output_units

        self.lstm = tf.keras.layers.LSTM(self.dec_units, return_state=True, return_sequences=True)
        self.dense_output = tf.keras.layers.Dense(self.output_units, activation=tf.nn.relu)

    def call(self, x, hidden=None, training=None):
        output, _, _ = self.lstm(x, initial_state=hidden, training=training)
        output = self.dense_output(output, training=training)
        return output

    def get_config(self):
        config = super(LSTMDecoder, self).get_config()
        config.update({"dec_units": self.dec_units, "output_units": self.output_units})
        return config


class LSTMAddonsDecoder(tf.keras.layers.Layer):
    def __init__(self, dec_units, output_units, name="lstm_addons_decoder", **kwargs):
        super(LSTMAddonsDecoder, self).__init__(name=name, **kwargs)
        self.dec_units = dec_units
        self.output_units = output_units

        # Dense output
        self.dense_output = tf.keras.layers.Dense(self.output_units, activation=tf.nn.relu)

        # Decoder
        self.sampler = tfa.seq2seq.TrainingSampler()
        self.decoder_cell = tf.keras.layers.LSTMCell(dec_units)
        self.decoder = tfa.seq2seq.BasicDecoder(self.decoder_cell, self.sampler, self.dense_output)

    def call(self, x, hidden=None, training=None):
        output, _, _ = self.decoder(x, initial_state=hidden, training=training)
        return output.rnn_output

    def get_config(self):
        config = super(LSTMAddonsDecoder, self).get_config()
        config.update({"dec_units": self.dec_units, "output_units": self.output_units})
        return config


class LSTMAutoregressiveDecoder(tf.keras.layers.Layer):
    def __init__(self, dec_units, output_steps, output_units, name="lstm_autoregressive_decoder", **kwargs):
        super(LSTMAutoregressiveDecoder, self).__init__(name=name, **kwargs)
        self.dec_units = dec_units
        self.output_steps = output_steps
        self.output_units = output_units

        # Autoregressive module
        self.lstm_cell = tf.keras.layers.LSTMCell(dec_units)
        # self.lstm = tf.keras.layers.RNN(self.lstm_cell, return_state=True)

        # Dense output
        self.dense_output = tf.keras.layers.Dense(self.output_units, activation=tf.nn.relu)

    def warmup(self, inputs, state, training=None):
        # input.shape => (batch, features)
        # prediction.shape => (batch, lstm_units)
        prediction, state = self.lstm_cell(inputs, states=state, training=training)

        # prediction.shape => (batch, features)
        prediction = self.dense_output(prediction, training=training)
        return prediction, state

    def call(self, last_input, hidden=None, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the lstm state
        x, state = self.warmup(last_input, hidden, training=training)
        # Insert the first prediction
        predictions.append(x)

        # Run the rest of the prediction steps
        for n in range(1, self.output_steps):
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state, training=training)
            x = self.dense_output(x, training=training)

            # Add the prediction to the output
            predictions.append(x)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions

    def get_config(self):
        config = super(LSTMAutoregressiveDecoder, self).get_config()
        config.update({"dec_units": self.dec_units, "output_steps": self.output_steps,
                       "output_units": self.output_units})
        return config


class AutoEncoder(tf.keras.Model):
    """
    config: "autoregressive", "addons", "simple"
    """
    def __init__(self, time_dim, feature_dim, latent_dim=1, name="autoencoder", **kwargs):
        super(AutoEncoder, self).__init__(name=name, **kwargs)
        self.time_dim = time_dim
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        self.encoder = LSTMEncoder(enc_units=latent_dim)
        self.decoder = None
        self.state = None

    def call(self, x, hidden=None, training=None):
        init_state = self.encoder.initialize_hidden_state()
        encoded, state_m, state_c = self.encoder(x, hidden=init_state, training=training)
        self.state = [state_m, state_c]
        decoded = self.decoder(encoded, hidden=self.state, training=training)
        return decoded

    def get_config(self):
        config = super(AutoEncoder, self).get_config()
        config.update({"time_dim": self.time_dim, "feature_dim": self.feature_dim,
                      "latent_dim": self.latent_dim})
        return config

    def train_step(self, data):
        # Unpack the data.
        x = data

        with tf.GradientTape() as tape:
            # Forward pass
            x_pred = self(x, training=True)
            # Compute the loss value
            loss = self.compiled_loss(x, x_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_weights
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(x, x_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def get_state(self):
        if not self.state:
            log.e("AutoEncoder state is None")
            return None
        return self.state


class RegressiveAutoEncoder(AutoEncoder):
    def __init__(self, time_dim, feature_dim, latent_dim=1, name="regressive_autoencoder", **kwargs):
        super(RegressiveAutoEncoder, self).__init__(time_dim, feature_dim, latent_dim, name, **kwargs)
        self.decoder = LSTMAutoregressiveDecoder(dec_units=self.latent_dim,
                                                 output_steps=self.time_dim,
                                                 output_units=self.feature_dim)

    def call(self, x, hidden=None, training=None):
        init_state = self.encoder.initialize_hidden_state()
        encoded, state_m, state_c = self.encoder(x, hidden=init_state, training=training)
        self.state = [state_m, state_c]
        decoded = self.decoder(x[:, -1, :], hidden=self.state, training=training)
        return decoded

    def train_step(self, data):
        return super(RegressiveAutoEncoder, self).train_step(data)


class AddonsAutoEncoder(AutoEncoder):
    def __init__(self, time_dim, feature_dim, latent_dim=1, name="addons_autoencoder", **kwargs):
        super(AddonsAutoEncoder, self).__init__(time_dim, feature_dim, latent_dim, name, **kwargs)
        self.decoder = LSTMAddonsDecoder(dec_units=self.latent_dim,
                                         output_units=self.feature_dim)

    def call(self, x, hidden=None, training=None):
        return super(AddonsAutoEncoder, self).call(x, hidden, training)

    def train_step(self, data):
        return super(AddonsAutoEncoder, self).train_step(data)


class SimpleAutoEncoder(AutoEncoder):
    def __init__(self, time_dim, feature_dim, latent_dim=1, name="addons_autoencoder", **kwargs):
        super(SimpleAutoEncoder, self).__init__(time_dim, feature_dim, latent_dim, name, **kwargs)
        self.decoder = LSTMDecoder(dec_units=self.latent_dim,
                                   output_units=self.feature_dim)

    def call(self, x, hidden=None, training=None):
        return super(SimpleAutoEncoder, self).call(x, hidden, training)

    def train_step(self, data):
        return super(SimpleAutoEncoder, self).train_step(data)


class DeepClustering:
    def __init__(self, data: pd.DataFrame, latent_dim, hidden_dim=None,
                 model="autoregressive", epoch=10, batch_sz=1, scale=True):
        # Tensorflow config
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

        # Attributes assignment
        self.seq_len = max(int(len(data.index) / 2), 1)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.model = model if model in ["autoregressive", "addons", "simple"] else "simple"
        self.epoch = epoch
        self.batch_sz = batch_sz
        self.features_len = len(data.columns)

        # Data
        if scale:
            data = (data - data.min()) / (data.max() - data.min())

        if len(data.index) > 1:
            self.data = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data.to_numpy(),
                targets=None,
                sequence_length=self.seq_len,
                sequence_stride=1,
                shuffle=True,
                batch_size=batch_sz)
            # self.data = self.data.map(lambda x: (x, x))
        else:
            data_np = np.expand_dims(data.values, axis=0)
            self.data = tf.data.Dataset.from_tensor_slices(data_np).batch(batch_sz)
        self.ae = None

    def train(self, patience=2):
        """
        optimizer = tf.keras.optimizers.Adam()
        mse_loss_fn = tf.keras.losses.MeanSquaredError()
        loss_metric = tf.metrics.MeanAbsoluteError()

        # Iterate over epochs.
        for epoch in range(self.epoch):
            log.i("Start of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            for step, x_batch_train in enumerate(self.data):
                with tf.GradientTape() as tape:
                    reconstructed = self.ae(x_batch_train)
                    # Compute reconstruction loss
                    loss = mse_loss_fn(x_batch_train, reconstructed)
                    loss += sum(self.ae.losses)  # Add KLD regularization loss

                    loss_metric(x_batch_train, reconstructed)

                grads = tape.gradient(loss, self.ae.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.ae.trainable_weights))

                # loss_metric(loss)

                log.i("step %d: loss %.4f; mae = %.4f" % (step, loss, loss_metric.result()))
        return
        """
        if self.model == "autoregressive":
            self.ae = RegressiveAutoEncoder(self.seq_len, self.features_len, latent_dim=self.latent_dim)
        elif self.model == "simple":
            self.ae = SimpleAutoEncoder(self.seq_len, self.features_len, latent_dim=self.latent_dim)
        elif self.model == "addons":
            self.ae = AddonsAutoEncoder(self.seq_len, self.features_len, latent_dim=self.latent_dim)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss",
                                                          patience=patience,
                                                          mode="min")
        self.ae.compile(loss=tf.keras.losses.MeanSquaredError(), run_eagerly=True,
                        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
        history = self.ae.fit(self.data, epochs=self.epoch, callbacks=[early_stopping])
        return history

    def get_train_state(self):
        train_loss, train_acc = self.ae.evaluate(self.data)
        return train_loss, train_acc

    def get_latent_state(self):
        state = self.ae.get_state()
        # Flat the state tuple of list.
        return pd.Series([s for sub in state for s in sub[-1].numpy()])
