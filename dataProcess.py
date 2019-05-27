import nltk
import os
import pickle
import glob
from gensim import corpora
import numpy as np
import tensorflow.contrib.keras as kr

class Preprocess:
    def __init__(self,file_dir,file_name,word_dict_file):

        self.getword_dict(word_dict_file,file_dir)
        self.process(file_dir,file_name)

    def get_s1(self):
        return self.s1

    def get_s2(self):
        return self.s2

    def get_labels(self):
        return self.label

    def get_s1_length(self):
        return self.s1_length

    def get_s2_length(self):
        return self.s2_length

    def get_word_dict(self):
        return self.word_dict

    def get_vocab_size(self):
        return self.word_size

    def get_data_sizw(self):
        return self.size

    def process(self, file_dir, file_name):
        file=os.path.join(file_dir,file_name)
        if not os.path.exists(file):
            print('%s is not exist!' %file)
            return
        self.sens1,self.sens2,label=self.getsence(file)
        self.label = np.array(label)
        self.size=len(self.label)

        self.s1_token=self.tokennize(self.sens1)
        s1=self.getid(self.s1_token)
        s1_length=self.getlength(s1)
        self.s2_token=self.tokennize(self.sens2)
        s2=self.getid(self.s2_token)
        s2_length=self.getlength(s2)

        self.s1 = np.array(s1)
        self.s2 = np.array(s2)
        self.s1_length = np.array(s1_length)
        self.s2_length = np.array(s2_length)


    def getlength(self, sentence):
        return [len(s) for s in sentence]

    def getid(self, sentence):
        return [[self.word_dict[word] for word in s]for s in sentence]

    def getword_dict(self,word_dict_file,file_dir):
        if os.path.exists(word_dict_file):
            print('word_dict file exist!')
            with open(word_dict_file,'rb') as fword:
                self.word_dict=pickle.load(fword)
            print('词典总数：',len(self.word_dict))
        else:
            self.word_dict=self.process_word_dict(file_dir,word_dict_file)
        self.word_size=len(self.word_dict)

    def process_word_dict(self,file_dir,word_dict_file):
        files=glob.glob(os.path.join(file_dir,'*.txt'))
        print(files)
        allsens=[]
        for i,f in enumerate(files):
                sens1,sens2,_=self.getsence(f)
                for s1, s2 in zip(sens1, sens2):
                    allsens.append(s1+" "+s2)

        allwords=self.tokennize(allsens)
        dict=corpora.Dictionary(allwords)
        words_dict=dict.token2id
        lenword=len(words_dict)
        words_dict['PADDING']=lenword
        words_dict['UNK']=lenword+1
        print("词典总数：",len(words_dict))
        print(words_dict)
        with open(word_dict_file,'wb')as f :
            pickle.dump(words_dict,f)
        return words_dict

    def tokennize(self,s):
        words = [nltk.word_tokenize(sen) for sen in s]
        return [[ w for w in word if w.strip()]for word in words]

    def getsence(self, f):
        sens1=[]
        sens2=[]
        labels=[]
        with open(f, 'r', encoding='utf-8') as fr:
            for i, lines in enumerate(fr):
                    l = lines.strip().split('\t')
                    sens1.append(l[0].strip('.'))
                    sens2.append(l[1].strip('.'))
                    labels.append(l[2])
        return sens1,sens2,labels


if __name__=='__main__':
    file_dir = '../data/prosciTail/'
    file_name = 'scitail_train.txt'
    word_dict_file = '../data/word_dict/sciTail_word_dict.pkl'
    data=Preprocess(file_dir,file_name,word_dict_file)

    print(data.size)

    print(data.sens1[1])
    print(data.s1[1])
    print(data.s1_length[1])

    print(data.sens2[:10])
    print(data.s2[1:10].tolist())
    print(data.s2_length[:10])
    print(data.label[:10 ])




