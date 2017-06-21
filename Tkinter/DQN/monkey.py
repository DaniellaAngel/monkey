from Tkinter import *  
import tkFont  
import os  
from functools import partial  
from PIL import Image, ImageTk  
import pandas as pd
import numpy as np
import Tkinter as tk
import random
import time

# dataset = pd.read_csv("dataset/GRF/JCUED03/enMatJCUED03_train.csv")
dataset = pd.read_csv("dataset/GRF/JUNIWARD03/enMatJUNIWARD03_train.csv")



# label = pd.read_csv("dataRL/train_label99.csv")
buttonNames = np.array(dataset.columns.values).astype('int')

# 99 0.8352
#

class Monkey(tk.Tk, object):
    """docstring for Monkey"""
    def __init__(self):
        super(Monkey, self).__init__()
        self.action_space = ['add','delete']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.objects = []
        self.REFERLEN = 61
        self.MAXLEN = 107
        self.REFERACC = 0.7967
        self.GOALACC = self.REFERACC
        # self.REFERLEN = 75
        # self.MAXLEN = 121
        # self.REFERACC = 0.8436
        # self.GOALACC = 0.8450
        self.title("Monkey")
        self.resizable(0,0)
        self.dataset = pd.read_csv("dataset/GRF/JCUED03/enMatJCUED03_train.csv")
        self.label = pd.read_csv("dataset/GRF/JCUED03/enMatJCUED03_train_label.csv")
        self.test_dataset = pd.read_csv("dataset/GRF/JCUED03/enMatJCUED03_test.csv")
        self.test_label = pd.read_csv("dataset/GRF/JCUED03/enMatJCUED03_test_label.csv")
        # self.dataset = pd.read_csv("dataset/GRF/JUNIWARD03/enMatJUNIWARD03_train.csv")
        # self.label = pd.read_csv("dataset/GRF/JUNIWARD03/enMatJUNIWARD03_train_label.csv")
        # self.test_dataset = pd.read_csv("dataset/GRF/JUNIWARD03/enMatJUNIWARD03_test.csv")
        # self.test_label = pd.read_csv("dataset/GRF/JUNIWARD03/enMatJUNIWARD03_test_label.csv")
        self.classifiers = pd.DataFrame()
        self.resetClassifier = pd.DataFrame()
        self._build_monkey()

    def _build_monkey(self):
        button_font = tkFont.Font(size=10, weight=tkFont.BOLD)  
        button_bg = '#D5E0EE'  
        button_active_bg = '#E5E35B'
        myButton = partial(Checkbutton, self, bg=button_bg, padx=10, pady=3, activebackground = button_active_bg)
        for item in buttonNames:
            button = myButton(text = item)
            button.grid(row=1, column=item, pady=5)
            if item > 10 and item< 21:
                button.grid(row=2, column=item-10, pady=5)
            if item > 20 and item < 31:
                button.grid(row=3, column=item-10*2, pady=5)
            if item > 30 and item < 41:
                button.grid(row=4, column=item-10*3, pady=5)
            if item > 40 and item < 51:
                button.grid(row=5, column=item-10*4, pady=5)
            if item > 50 and item < 61:
                button.grid(row=6, column=item-10*5, pady=5)
            if item > 60 and item < 71:
                button.grid(row=7, column=item-10*6, pady=5)
            if item > 70 and item < 81:
                button.grid(row=8, column=item-10*7, pady=5)
            if item > 80 and item < 91:
                button.grid(row=9, column=item-10*8, pady=5)
            if item > 90 and item < 101:
                button.grid(row=10, column=item-10*9, pady=5)
            if item > 100 and item< 111:
                button.grid(row=11, column=item-10*10, pady=5)
            if item > 110 and item< 121:
                button.grid(row=12, column=item-10*11, pady=5)
            if item > 120 and item< 131:
                button.grid(row=13, column=item-10*12, pady=5)
            if item > 130 and item< 141:
                button.grid(row=14, column=item-10*13, pady=5)
            if item > 140 and item< 151:
                button.grid(row=15, column=item-10*14, pady=5)
            self.objects.append(button)

        entry_font = tkFont.Font(size=12)  
        entry = Entry(self, justify="right", font=entry_font)
        entry.grid(row=0, column=0, columnspan=11, sticky=N+W+S+E, padx=5,  pady=5)
        self.objects.append(entry)

        goalButton = Button(self,text="Goal",bg = "blue",fg = "white")
        goalButton.grid(row=16,column=0,columnspan=5,sticky=N+W+S+E,padx=5,pady=5)
        self.objects.append(goalButton)

        fireButton = Button(self,text="Fire",bg = "blue",fg = "white")
        fireButton.grid(row=16,column=5,columnspan=5,sticky=N+W+S+E,padx=5,pady=5)
        self.objects.append(fireButton)



        # self.mainloop()

    def reset(self):
        self.classifiers = self.resetClassifier
        self.init_classifiers = np.array(random.sample(list(self.dataset.columns.values),11)).astype('int')
        for item in self.init_classifiers:
            self.objects[item-1].select()
        self.initones = np.hstack((np.array([1]),)*11)
        self.classifiers[self.init_classifiers] = self.dataset.iloc[:,self.init_classifiers-self.initones]
        self.initpred_labels = self.pred_label()
        self.init_accuracy = self.caculate_acc(self.initpred_labels)
        
        self.init_state = np.array([len(self.classifiers.columns.values),self.init_accuracy])

        return self.init_state

    def pred_label(self):
        if len(self.classifiers.values) == 0:
            print "initial classifiers"
            self.reset()
            return
        else:
            self.add_labels = self.classifiers.apply(lambda x: x.sum(), axis=1).values
            self.pred_labels = []
            for item1 in self.add_labels:
                if item1 > 0:
                    self.pred_labels.append(1)
                else:
                    self.pred_labels.append(-1)
            return self.pred_labels

    def caculate_acc(self,pred_value):
        diff_labels = pred_value - self.label.values.T[0]
        count = 0
        for item2 in diff_labels:
            if item2 == 0:
                count+=1
            else:
                count = count
        accuracy = float(count)/len(self.label.values.T[0])

        return accuracy

    def step(self,action):
        if len(self.classifiers.columns.values)%2 == 0:
                dropn01 = random.sample(list(self.classifiers.columns.values),1)
                for item01 in dropn01:
                    self.objects[item01-1].deselect()
                self.classifiers.drop(dropn01,axis=1,inplace=True)
        if len(self.classifiers.columns.values) == 10 or len(self.classifiers.columns.values) < 10:
            self.init_classifiers = np.array(random.sample(list(self.dataset.columns.values),11)).astype('int')
            for item in self.init_classifiers:
                self.objects[item-1].select()
            self.initones = np.hstack((np.array([1]),)*11)
            self.classifiers[self.init_classifiers] = self.dataset.iloc[:,self.init_classifiers-self.initones]
        
        pred_labels0 = self.pred_label()
        accuracy0 = self.caculate_acc(pred_labels0)
        s = np.array([len(self.classifiers.columns.values),accuracy0])
        midCalssifiers = self.classifiers
        if action == 0:
            arr0 = np.array(random.sample(list(self.dataset.columns.values),5)).astype('int')
            arrnd = np.hstack((np.array([1]),)*5)
            for item in arr0:
                self.objects[item-1].select()
            self.classifiers[arr0] = self.dataset.iloc[:,arr0-arrnd]

            if len(self.classifiers.columns.values)%2 == 0:
                dropn02 = random.sample(list(self.classifiers.columns.values),1)
                for item02 in dropn02:
                    self.objects[item02-1].deselect()
                self.classifiers.drop(dropn02,axis=1,inplace=True)
            # print "action0",len(self.classifiers.columns.values)
            pred_labels = self.pred_label()
            accuracy = self.caculate_acc(pred_labels)
            self.objects[-3].delete(0,END)
            if accuracy > accuracy0:
                s_ = np.array([len(self.classifiers.columns.values),accuracy])
            else:
                self.classifiers = midCalssifiers
                s_ = s
            self.objects[-3].insert(END,s_)
            #self.objects[-3].insert(0,accuracy)
            #self.objects[-3].insert(1,len(self.classifiers.columns.values))
        elif action == 1:
            dropnN = random.sample(list(self.classifiers.columns.values),3)
            for item in dropnN:
                self.objects[item-1].deselect()
            self.classifiers.drop(dropnN,axis=1,inplace=True)

            if len(self.classifiers.columns.values)%2 == 0:
                dropn1 = random.sample(list(self.classifiers.columns.values),1)
                for item03 in dropn1:
                    self.objects[item03-1].deselect()
                self.classifiers.drop(dropn1,axis=1,inplace=True)
            # print "action1",len(self.classifiers.columns.values)
            pred_labels = self.pred_label()
            accuracy = self.caculate_acc(pred_labels)
            self.objects[-3].delete(0,END)
           
            if accuracy > accuracy0:
                s_ = np.array([len(self.classifiers.columns.values),accuracy])
            else:
                self.classifiers = midCalssifiers
                s_ = s
            self.objects[-3].insert(END,s_)
            # self.objects[-3].insert(1,len(self.classifiers.columns.values))

        if (s_[1] > self.GOALACC or s_[1] == self.GOALACC) and s_[0] > self.REFERLEN:
            self.objects[-2].config(bg = "green")
            reward = 1
            done = True
            self.GOALACC = s_[1]
            self.resetClassifier = self.classifiers
            print "Goal lists====>",self.classifiers.columns.values
            print "Acc=====>",s_[1]
            print "length====>",len(self.classifiers.columns.values)
        else :
            self.objects[-1].config(bg = "red")
            reward = 0
            done = False
       	#self.objects[-2].config(bg = "blue")
       	#self.objects[-1].config(bg = "blue")
        result = self.classifiers.columns.values
        return s_,reward,done,result
    def render(self):
        time.sleep(0.1)
        self.update()

# env = Monkey()

