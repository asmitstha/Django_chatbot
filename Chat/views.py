from django.shortcuts import render,redirect
from django.http import HttpResponse
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras import layers, activations, models, preprocessing
from keras.callbacks import ModelCheckpoint
import requests, zipfile, io
from keras.preprocessing.text import Tokenizer
from tensorflow.keras import preprocessing, utils
from keras.regularizers import l1
import os, sys
import yaml
import re
import string
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login
from .models import Product

def home(request):
    products = Product.objects.all()
    return render(request,"home.html", {'products':products})

def index(request):
    products = Product.objects.all()
    return render(request,"index.html", {'products':products})


def single(request, slug):
    product = Product.objects.get(slug=slug)
    return render(request,"single_product.html", {'product':product})

def register(request):
    if request.method == 'POST':
        form= UserCreationForm(request.POST)

        if form.is_valid():
            form.save()
            username = form.cleaned_data['username']
            password = form.cleaned_data['password1']

            user=authenticate(username=username, password=password)
            login(request, user)
            return redirect('index')
    else:
        form =UserCreationForm()

    context={'form':form}
    return render(request,'registration/register.html',context)



path = r"data"
files_list = os.listdir(path)
try:
    model.load_weights(checkpoint_path)
except:
    questions = list()
    answers = list()

    for filepath in files_list:
        stream = open(path + os.sep + filepath, 'rb')

        docs = yaml.safe_load(stream)
        conversations = docs['conversations']
        for con in conversations:
            if len(con) > 2:
                questions.append(con[0])
                replies = con[1:]
                ans = ''
                for rep in replies:
                    ans += ' ' + rep
                answers.append(ans)
            elif len(con) > 1:
                questions.append(con[0])
                answers.append(con[1])

    answers_with_tags = list()
    for i in range(len(answers)):
        if type(answers[i]) == str:
            answers_with_tags.append(answers[i])
        else:
            questions.pop(i)

    answers = list()
    for i in range(len(answers_with_tags)):
        answers.append('<START> ' + answers_with_tags[i] + ' <end>')

    tokenizer = preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(questions + answers)

    VOCAB_SIZE = len(tokenizer.word_index) + 1
    # print( 'VOCAB SIZE : {}'.format( VOCAB_SIZE ))

    vocab = []
    for word in tokenizer.word_index:
        vocab.append(word)


    def tokenize(sentences):
        tokens_list = []
        vocabulary = []
        for sentence in sentences:
            sentence = str(sentence).lower()
            sentence = re.sub('[^a-zA-Z]', ' ', sentence)
            result = sentence.translate(str.maketrans("", "", string.punctuation))
            tokens = result.split()
            vocabulary += tokens
            tokens_list.append(tokens)
        return tokens_list, vocabulary


    # encoder_input_data
    tokenized_questions = tokenizer.texts_to_sequences(questions)  # Transform each question in a sequence of integers.
    maxlen_questions = max([len(x) for x in tokenized_questions])  # question with maximum length i.e is 50
    padded_questions = preprocessing.sequence.pad_sequences(tokenized_questions, maxlen=maxlen_questions,
                                                            padding='post')  # ensure that all sequences in a list have the same length
    encoder_input_data = np.array(padded_questions)  # create array

    # decoder_input_data
    tokenized_answers = tokenizer.texts_to_sequences(answers)  # Transform each text in texts in a sequence of integers.
    maxlen_answers = max([len(x) for x in tokenized_answers])  # answer with maximum length i.e is 74
    padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers,
                                                          padding='post')  # ensure that all sequences in a list have the same length
    decoder_input_data = np.array(padded_answers)  # create array

    # decoder_output_data: Tokenize the answers. Remove the first element from all the tokenized_answers.
    # This is the <START> element which we added earlier.
    tokenized_answers = tokenizer.texts_to_sequences(answers)  # Transform each text in texts in a sequence of integers.
    for i in range(len(tokenized_answers)):
        tokenized_answers[i] = tokenized_answers[i][1:]  # remove first element form all the tokenized_answers.

    padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers,
                                                          padding='post')  # ensure that all sequences in a list have the same length
    onehot_answers = utils.to_categorical(padded_answers,
                                          VOCAB_SIZE)  # Converts a class vector (integers) to binary class matrix.
    decoder_output_data = np.array(onehot_answers)  # create array of onehot_answers

    encoder_inputs = tf.keras.layers.Input(shape=(None,))  # Input() is used to instantiate a Keras tensor
    encoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 300, mask_zero=True)(
        encoder_inputs)  # Turns positive integers (indexes) into dense vectors of fixed size
    encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(300, return_state=True)(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = tf.keras.layers.Input(shape=(None,))
    decoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 300, mask_zero=True)(decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(300,activity_regularizer=l1(0.001), return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = tf.keras.layers.Dense(VOCAB_SIZE, activation=tf.keras.activations.softmax)
    output = decoder_dense(decoder_outputs)

    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True
                                                     )

try:
    model.load_weights(checkpoint_path)
except:
    model.fit([encoder_input_data, decoder_input_data], decoder_output_data, batch_size=64, epochs=40,
              callbacks=[cp_callback])


def make_inference_models():
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

    decoder_state_input_h = tf.keras.layers.Input(shape=(300,))
    decoder_state_input_c = tf.keras.layers.Input(shape=(300,))

    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model


def str_to_tokens(sentence: str):
    words = sentence.lower().split()
    words = sentence.translate(str.maketrans("", "", string.punctuation)).split()
    tokens_list = list()

    for word in words:
        if word in vocab:
            tokens_list.append(tokenizer.word_index[word])
        elif word == "quit":
            print("See ya!!")

        else:
            return str_to_tokens("understand")
            # reply =  "haha what??"
            # decoded_translation =
            # decoded_translation = "BOT :" + ''
            # #print("BOT : Can you eloborate the word {}.".format(word))
            break
    return preprocessing.sequence.pad_sequences([tokens_list], maxlen=maxlen_questions, padding='post')

def chatbot(request):
    enc_model, dec_model = make_inference_models()

    while True:
        val1 = request.GET["msg"]
        states_values = enc_model.predict(str_to_tokens(val1))
        if val1 == 'quit':
            break

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = tokenizer.word_index['start']
        stop_condition = False
        decoded_translation = "BOT :" + ''
        while not stop_condition:
            dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)
            sampled_word_index = np.argmax(dec_outputs[0, -1, :])
            sampled_word = None
            for word, index in tokenizer.word_index.items():
                if sampled_word_index == index:
                    decoded_translation += ' {}'.format(word)
                    sampled_word = word

            if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
                stop_condition = True

            empty_target_seq = np.zeros((1, 1))
            empty_target_seq[0, 0] = sampled_word_index
            states_values = [h, c]

        return HttpResponse(decoded_translation)


# def chatbot(request):
#     # return HttpResponse("hello all")
#     enc_model, dec_model = make_inference_models()
#
#     while True:
#         # take = input('YOU : ')
#         val1 = request.GET["msg"]
#         states_values = enc_model.predict(str_to_tokens(val1))
#         if val1 == 'quit':
#             break
#
#         empty_target_seq = np.zeros((1, 1))
#         empty_target_seq[0, 0] = tokenizer.word_index['start']
#         stop_condition = False
#         decoded_translation =''
#         while not stop_condition:
#             dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)
#             sampled_word_index = np.argmax(dec_outputs[0, -1, :])
#             sampled_word = None
#             for word, index in tokenizer.word_index.items():
#                 if sampled_word_index == index:
#                     decoded_translation += ' {}'.format(word)
#                     sampled_word = word
#
#             if sampled_word == '<end>' or len(decoded_translation.split()) > maxlen_answers:
#                 stop_condition = True
#
#             empty_target_seq = np.zeros((1, 1))
#             empty_target_seq[0, 0] = sampled_word_index
#             states_values = [h, c]
#
#         # return decoded_translation
#         return HttpResponse(decoded_translation)
#         # print(decoded_translation)
