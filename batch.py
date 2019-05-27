import numpy as np

from dataProcess import Preprocess
import tensorflow.contrib.keras as kr


class BatchGenerator(object):
    '''
    construct raw_data generator.The input X,y should be narray or list like type
    '''
    def __init__(self,X1,X2,y,x1_len,x2_len,word_dict,shuffle):
        if type(X1)!=np.ndarray:         #为什么是ndarray类型？？
            X=np.array(X1)
        if type(y)!=np.ndarray:
            y=np.array(y)
        self.X1=X1
        self.X2=X2
        self.y=y
        self.x1_len=x1_len
        self.x2_len=x2_len
        self.word_dicts=word_dict
        self._epochs_completed=0
        self._index_in_epoch=0
        self._number_example=X1.shape[0]
        self.shuffle=shuffle
        self.padnum = self.word_dicts['PADDING']

        self.datasize=len(y)
        if shuffle:                                                         #True 为先提前打乱好
            index=np.random.permutation(self._number_example)
            self.X1=self.X1[index]
            self.X2=self.X2[index]
            self.x1_len=self.x1_len[index]
            self.x2_len = self.x2_len[index]
            self.y=self.y[index]

    def X(self):
            return self.X

    def y(self):
            return self.y

    def get_size(self):
        return self.datasize

    def _epochs_completed(self):
            return self._epochs_completed

    def _number_example(self):
            return self._number_example

    def padding(self, sentences,max_len):

        sens=[]
        # 这里长的要截掉！！设计引用的问题，，
        for s in sentences:
                num = max_len - len(s)
                # 这里的s0需要单独用切片操作取出，否则还是在原list上操作
                # 还有一种方法是将长的截取！
                s0 = s[:]
                for i in range(num):
                    s0.append(self.word_dicts['PADDING'])
                sens.append(s0)
        return sens

    def next_batch(self,batch_size):
            '''return raw_data in batch_size
                consider epoche
            '''
            start=self._index_in_epoch
            self._index_in_epoch+=batch_size
            if self._index_in_epoch>self._number_example:
                self._epochs_completed+=1
                if self.shuffle:
                    index = np.random.permutation(self._number_example)
                    self.X1 = self.X1[index]
                    self.X2 = self.X2[index]
                    self.x1_len = self.x1_len[index]
                    self.x2_len = self.x2_len[index]
                    self.y = self.y[index]

                start=0                                                      #这里这么写是因为又开始了新的batch，start=0开始
                self._index_in_epoch=batch_size
                assert batch_size<self._number_example
            end=self._index_in_epoch

            s1_len = self.x1_len[start:end]
            s2_len=self.x2_len[start:end]
            label=self.y[start:end]
            max_s1_len= max(s1_len)
            max_s2_len=max(s2_len)
            s1_random=self.X1[start:end]
            s2_random=self.X2[start:end]

            s1 = kr.preprocessing.sequence.pad_sequences(s1_random, max_s1_len, padding='post', value=self.padnum)
            s2 = kr.preprocessing.sequence.pad_sequences(s2_random, max_s2_len, padding='post', value=self.padnum)

            sen1_mask = (s1 != self.padnum).astype(np.int32)
            sen2_mask = (s2 != self.padnum).astype(np.int32)

            return s1,s2,label,s1_len,s2_len,max_s1_len,max_s2_len,sen1_mask,sen2_mask,

if __name__=='__main__':
    file_dir2 = '../data/prosciTail/'
    file_name2 = 'scitail_dev.txt'
    word_dict_file = '../data/word_dict/sciTail_word_dict.pkl'

    # train=DataHelper(file_dir2,file_name2,word_dict_file,20,True)
    data = Preprocess(file_dir2, file_name2, word_dict_file)
    sentence1 = data.s1
    sentence2 = data.s2
    label = data.label
    sen1_length = data.s1_length
    sen2_length = data.s2_length
    word_dict = data.word_dict

    data_train = BatchGenerator(sentence1, sentence2, label, sen1_length, sen2_length, word_dict, True)

    for i in range(1):
        s1, s2, label, s1_len, s2_len, max_len,s1_mask,s2_mask = data_train.next_batch(2)
        print(s1)
        print(s1_len)
        print(s1_mask)

        print(s2)
        print(s2_len)
        print(s2_mask)
        print(label)
        print(max_len)

