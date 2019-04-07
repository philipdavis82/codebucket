from __future__ import absolute_import, division, print_function

import tensorflow as tf

import cookie_parser
import numpy as np
import pandas as pd
import os,time

class data():
    def __init__(self):
        tf.enable_eager_execution()
        self.get_fortunes()
        self.find_vocab()
        self.gen_vocab_maps()
        self.create_training_examples()
        self.batch_packing()


    def get_fortunes(self):
        parser = cookie_parser.parser()
        try:
            self.fortunes = parser.load_from_main_data()
        except:
            parser.load_from_repo_data()
            parser.save()
            self.fortunes = parser.load_from_main_data()
        parser = None

    def find_vocab(self):
        self.text = self.fortunes

        #print(self.text[:50])
        self.vocab = sorted(set(self.text))
    
    def gen_vocab_maps(self):
        self.char2idx = {u:i for i, u in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)
        
        self.fortunes_as_int = np.array([self.char2idx[c] for c in self.text])
    
    def create_training_examples(self):
        self.seq_length = 100
        self.examples_per_epoch = len(self.text)//self.seq_length
        self.char_dataset = tf.data.Dataset.from_tensor_slices(self.fortunes_as_int)
        self.sequences = self.char_dataset.batch(self.seq_length+1,drop_remainder=True)
        self.dataset = self.sequences.map(self._split_input_target)
        

        for input_example,target_example in self.dataset.take(1):
            print ('Input data: ', repr(''.join(self.idx2char[input_example.numpy()])))
            print ('Target data:', repr(''.join(self.idx2char[target_example.numpy()])))

    def batch_packing(self):
        self.BATCH_SIZE = 64
        self.steps_per_epoch = self.examples_per_epoch//self.BATCH_SIZE
        self.BUFFER_SIZE = 10000
        self.dataset = self.dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True)
        print(self.dataset)

    def _split_input_target(self,chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text,target_text


class learner():
    def __init__(self):
        self.checkpoint_dir = './training_checkpoints'
        self.EPOCHS = 3

        self.data = data()
        self.model_setup()
        self.model = self._build_model(
                            vocab_size = self.vocab_size,
                            embedding_dim = self.embedding_dim,
                            rnn_units = self.rnn_units,
                            batch_size = self.data.BATCH_SIZE)
        self.setup_checkpoints()
        #self.train_model()
        #self.load_model()
        #print(self.generate_text(u"Don't "))

    def load_model_for_training(self):
        self.model = self._build_model(
                            vocab_size = self.vocab_size,
                            embedding_dim = self.embedding_dim,
                            rnn_units = self.rnn_units,
                            batch_size = self.data.BATCH_SIZE)
        
        self.model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))

        self.model.build(tf.TensorShape([1, None]))

    def model_setup(self):
        self.vocab_size = len(self.data.vocab)
        self.embedding_dim = 256
        self.rnn_units = 1024
        if tf.test.is_gpu_available():
            self.rnn = tf.keras.layers.CuDNNGRU
        else:
            import functools
            self.rnn = functools.partial(tf.keras.layers.GRU,recurrent_activation='sigmoid')

    def run_model(self):
        for input_example_batch, target_example_batch in self.data.dataset.take(1): 
            example_batch_predictions = self.model(input_example_batch)
            print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
        self.sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
        self.sampled_indices = tf.squeeze(self.sampled_indices,axis=-1).numpy()
        print("Input: \n", repr("".join(self.data.idx2char[input_example_batch[0]])))
        print()
        print("Next Char Predictions: \n", repr("".join(self.data.idx2char[self.sampled_indices ])))

    def train_model(self):
        def loss(labels,logits):
            return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        self.model.compile(
            optimizer=tf.train.AdamOptimizer(),
            loss=loss
        )
        self.history = self.model.fit(
            self.data.dataset.repeat(), 
            epochs=self.EPOCHS, 
            steps_per_epoch=self.data.steps_per_epoch, 
            callbacks=[self.checkpoint_callback])

    def load_model(self):
        self.model = self._build_model(
                            vocab_size = self.vocab_size,
                            embedding_dim = self.embedding_dim,
                            rnn_units = self.rnn_units,
                            batch_size = 1)

        self.model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))

        self.model.build(tf.TensorShape([1, None]))

    def get_text(self,string,temp=1,length=1000):
        self.num_generate = 1000
        self.temperature = 1.0
        print(self.generate_text(string))

    def generate_text(self,start_string):
         # Evaluation step (generating text using the learned model)

        # Number of characters to generate
        

        # Converting our start string to numbers (vectorizing) 
        input_eval = [self.data.char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # Empty string to store our results
        text_generated = []

        # Low temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        # Experiment to find the best setting.
        

        # Here batch size == 1
        self.model.reset_states()
        for i in range(self.num_generate):
            predictions = self.model(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            # using a multinomial distribution to predict the word returned by the model
            predictions = predictions / self.temperature
            predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
            
            # We pass the predicted word as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)
            
            text_generated.append(self.data.idx2char[predicted_id])

        return (start_string + ''.join(text_generated))

    def setup_checkpoints(self):
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt_{epoch}")

        self.checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_prefix,
            save_weights_only=True)

    def _build_model(self,vocab_size,embedding_dim,rnn_units,batch_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size,embedding_dim,
                                    batch_input_shape=[batch_size,None]),
            self.rnn(rnn_units,
                return_sequences=True,
                recurrent_initializer='glorot_uniform',
                stateful=True),
            tf.keras.layers.Dense(vocab_size)
        ])
        return model

