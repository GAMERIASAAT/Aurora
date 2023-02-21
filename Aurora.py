import tensorflow as tf
import tensorflow_datasets as tfds
import nltk
from nltk.tokenize import word_tokenize
import numpy as np

class Chatbot:
    def __init__(self):
        # Load the pre-trained model and tokenizer from disk
        self.model = self.build_model()
        self.tokenizer = self.load_tokenizer()
        # Set the maximum length of the input sequence for the model
        self.max_length = 20

    def load_tokenizer(self):
        # Load the pre-trained tokenizer from disk
        tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file('tokenizer')
        return tokenizer

    def build_model(self):
        # Define the neural network architecture using Keras API
        embedding_dim = 256
        rnn_units = 1024

        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.tokenizer.vocab_size, embedding_dim),
            tf.keras.layers.LSTM(rnn_units),
            tf.keras.layers.Dense(self.tokenizer.vocab_size, activation='softmax')
        ])
        # Load the pre-trained weights of the model from disk
        model.load_weights('model_weights')
        return model

    def preprocess_input(self, text):
        # Preprocess the user input by tokenizing and removing non-alphabetic characters
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token.isalpha()]
        return ' '.join(tokens)

    def generate_response(self, user_input):
        # Encode the preprocessed user input using the tokenizer
        input_tokens = self.tokenizer.encode(self.preprocess_input(user_input))
        # Limit the input sequence to the maximum length
        input_tokens = input_tokens[-self.max_length:]
        input_seq = tf.expand_dims(input_tokens, 0)

        temperature = 0.7
        output = self.model.predict(input_seq)
        output = output[0, -1, :] / temperature
        output = tf.random.categorical(output, num_samples=1)
        output = output[-1, 0].numpy()

        response_tokens = [output]
        for i in range(100):
            input_seq = tf.expand_dims(response_tokens, 0)
            output = self.model.predict(input_seq)
            output = output[0, -1, :] / temperature
            output = tf.random.categorical(output, num_samples=1)
            output = output[-1, 0].numpy()

            # Stop generating tokens if the end-of-string token is generated
            if output == self.tokenizer.vocab_size - 1:
                break

            response_tokens.append(output)

        response_text = self.tokenizer.decode(response_tokens).strip()
        return response_text.capitalize()

    def get_feedback(self, user_input, correct_response):
        # Prompt the user for feedback if the chatbot's response is incorrect or if it doesn't know how to respond
        feedback = input(f"Sorry, I don't know how to respond to '{user_input}'. Can you provide the correct response? ")
        # If the user provides the correct response, update the model's weights
        if feedback:
            target_tokens = self.tokenizer.encode(correct_response)
            target_seq = tf.expand_dims(target_tokens, 0)
            input_tokens = self.tokenizer.encode(user_input)
            input_seq = tf.expand_dims(input_tokens, 0)

            loss = 0
            with tf.GradientTape() as tape:
                output = self.model(input_seq)
                loss = tf.keras.losses.sparse_categorical_crossentropy(target_seq, output)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer = tf.keras.optimizers.Adam()
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            print(f"Model updated with feedback: '{feedback}'")
        return feedback

    def run(self):
        # Start the chatbot by prompting the user for input
        print('Hi! I am a chatbot. How can I help you?')
        while True:
            user_input = input('> ')
            if user_input == 'exit':
                break
            response = self.generate_response(user_input)
            # Check if the response is correct by prompting the user for feedback
            while response.lower() != 'exit':
                print(response)
                correct_response = input("Was my response correct? If not, please provide the correct response or type 'exit' to cancel. ")
                if correct_response.lower() == 'exit':
                    break
                elif correct_response:
                    feedback = self.get_feedback(user_input, correct_response)
                    if feedback:
                        # Re-generate the response using the updated model weights
                        self.model = self.build_model()
                        response = self.generate_response(user_input)
                    else:
                        # If the feedback is empty, use the previous response
                        response = feedback
                        continue
                else:
                    # If the correct response is empty, use the previous response
                    response = correct_response
                    continue

        print('Goodbye!')
