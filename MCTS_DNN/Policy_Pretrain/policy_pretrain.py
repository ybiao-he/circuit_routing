#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.models import load_model

import numpy as np
import os
import core_keras as core
import random
import sys

class learnPolicy(object):

    def __init__(self, filename):

        self.filename = filename

        self.boards = []
        self.labels = []

        self.data = []

        self.train_boards = []
        self.train_labels = []
        self.test_boards = []
        self.test_labels = [] 

        self.policy_model = None

        self.predict = []

    def readData(self):

    	self.data = np.genfromtxt(self.filename, delimiter=',')

        # for filename in os.listdir(self.foldername):
        #     path = self.foldername + filename
        #     self.data.append( np.genfromtxt(path, delimiter=',') )

    def splitBoardLabel(self):

        # random.shuffle(self.data)

        self.boards = np.array(self.data)[:,:-1]
        self.labels = np.array(self.data)[:,-1]

        self.labels = to_categorical(self.labels)

        self.boards = np.reshape(self.boards, (self.boards.shape[0], 30, 30, 1))

    def splitTrainTest(self, ratio):

        num_data = len(self.boards)
        self.train_boards = self.boards[:int(num_data*ratio)]
        self.train_labels = self.labels[:int(num_data*ratio)]

        self.test_boards = self.boards[int(num_data*ratio):]
        self.test_labels = self.labels[int(num_data*ratio):]


    def train(self):

        self.readData()
        self.splitBoardLabel()
        self.splitTrainTest(0.8)
        self.policy_model = core.actor_cnn()

        # train cnn
        self.policy_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

        # train mlp
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
        # self.policy_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        # self.policy_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        self.policy_model.fit(self.train_boards, self.train_labels, epochs=20, batch_size=200, 
            validation_data=(self.test_boards, self.test_labels))

        self.policy_model.save("actor.h5")

    def evaluation(self, features, truth):

        self.predict = self.policy_model.predict(features)

        errors = np.argmax(self.predict, axis=1) - np.argmax(truth, axis=1)

        num_correct = 0
        for e in errors:
            if np.count_nonzero(e) == 0:
                num_correct += 1
        print(num_correct/errors.shape[0])
        print(self.predict)
        # np.savetxt('results.csv', self.predict, delimiter=',')
        print(np.argmax(truth, axis=1))
        # np.savetxt('true.csv', truth, delimiter=',')

if __name__== "__main__":

    training_file = sys.argv[1]
    state_encoder = learnPolicy(training_file)

    state_encoder.train()

    state_encoder.evaluation(state_encoder.test_boards, state_encoder.test_labels)