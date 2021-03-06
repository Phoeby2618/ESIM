import tensorflow as tf
from tensorflow.contrib.layers import batch_norm,l2_regularizer
from module import word_match,mask_softmax



class DecompAtt:
    def __init__(self,args):

        self.embeddings = args.word_embedding
        self.vocab_size = args.vocab_size
        self.embedding_dim=args.word_embed_size
        self.hidden_size=args.hidden_size
        self.n_class = args.n_class

        self.is_train = args.is_train
        self.keep_drop = args.drop_keep_prob
        self.l2_reg=args.l2_reg
        self.encoder_shared=args.encoder_shared
        self.infer_shared=args.infer_shared
        self.optimizer=args.optimizer
        self.clip_value=args.clip_value

        self.bulit_graph()

    def bulit_graph(self):
        self._placeholder_init()
        self._embedding()

        self._attend()
        self._compare()
        self._aggregate()
        self._pred()

        self.accu()
        self.loss_op()
        self.train_op()


    def _placeholder_init(self):
        '''
        max_s1_len:max x len of one batch
        max_s2_len:max y len of one batch
        :return:
        '''
        self.x = tf.placeholder(tf.int32, [None, None], name="x")
        self.y = tf.placeholder(tf.int32, [None, None], name="y")

        self.x_mask = tf.placeholder(tf.float32, [None, None], name="x_mask")
        self.y_mask = tf.placeholder(tf.float32, [None, None], name="y_mask")

        self.x_len = tf.placeholder(tf.int32, [None,], name="x_len")
        self.y_len = tf.placeholder(tf.int32, [None, ], name="y_len")

        self.max_s1_len = tf.placeholder(dtype=tf.int32, shape=[], name='max_s1_len')
        self.max_s2_len = tf.placeholder(dtype=tf.int32, shape=[], name='max_s2_len')

        self.label = tf.placeholder(tf.int32, [None,self.n_class], name="label")
        self.learning_rate=tf.placeholder(tf.float32,None,name='learning_rate')

    def _embedding(self):
        with tf.variable_scope('word_embedding'):
            if self.embeddings is None:
                print('embedding random!')
                self.embeddings=tf.get_variable('random_embeddings',shape=[self.vocab_size,self.embedding_dim],
                                                dtype=tf.float32,initializer=tf.random_normal_initializer())
            else :
                print('use word embedding!')
                self.embeddings=tf.convert_to_tensor(self.embeddings,dtype=tf.float32)
            self.x_embed=tf.nn.embedding_lookup(self.embeddings,self.x)
            self.y_embed=tf.nn.embedding_lookup(self.embeddings,self.y)
            if self.is_train and self.keep_drop:
                self.x_embed=tf.nn.dropout(self.x_embed,self.keep_drop)
                self.y_embed = tf.nn.dropout(self.y_embed, self.keep_drop)

    def _attend(self):
        with tf.variable_scope('attend'):
            self.x_projection=self.linear(self.x_embed,self.hidden_size*2,'xW_pro','xb_pro',
                                  activation=tf.nn.relu,regularizar=l2_regularizer(self.l2_reg))
            self.y_projection = self.linear(self.y_embed, self.hidden_size * 2, 'yW_pro', 'yb_pro',
                                            activation=tf.nn.relu, regularizar=l2_regularizer(self.l2_reg))


            att_mat=word_match(self.x_projection,self.y_projection)
            x_mask=tf.tile(tf.expand_dims(self.x_mask,axis=1),(1,self.max_s2_len,1))
            y_mask=tf.tile(tf.expand_dims(self.y_mask,axis=1),(1,self.max_s1_len,1))
            soft_y=mask_softmax(att_mat,y_mask)
            soft_x=mask_softmax(tf.transpose(att_mat,perm=[0,2,1]),x_mask)

            self.x_wy=tf.matmul(soft_y,self.y_projection)
            self.y_wx=tf.matmul(soft_x,self.x_projection)

    def _compare(self):
        with tf.variable_scope('compare'):
            x_fea=tf.concat([self.x_projection,self.x_wy],axis=-1)
            y_fea = tf.concat([self.y_projection, self.y_wx], axis=-1)

            if self.is_train and self.keep_drop:
                x_fea=tf.nn.dropout(x_fea,self.keep_drop)
                y_fea = tf.nn.dropout(y_fea, self.keep_drop)

            self.x_com=self.linear(x_fea,self.hidden_size*2,'xW_com','xb_com',
                                   activation=tf.nn.relu,regularizar=l2_regularizer(self.l2_reg))
            self.y_com = self.linear(y_fea, self.hidden_size * 2, 'yW_com','yb_com',
                                     activation=tf.nn.relu,regularizar=l2_regularizer(self.l2_reg))


    def _aggregate(self):
        with tf.variable_scope('aggregate'):
            v1=tf.reduce_mean(self.x_com,axis=1)
            v2=tf.reduce_mean(self.y_com,axis=1)
            self.v=tf.concat([v1,v2],axis=-1)

            if self.is_train and self.keep_drop<1:
                self.v=tf.nn.dropout(self.v,keep_prob=self.keep_drop)

    def _pred(self):
        with tf.variable_scope('predict'):
            output=self.linear(self.v,self.hidden_size*2,'fnn_W','fnn_b',
                               activation=tf.nn.relu,regularizar=l2_regularizer(self.l2_reg))

            self.pred=tf.nn.softmax(self.linear(output,self.n_class,'w_pre'))

    def accu(self):
        correct=tf.equal(tf.argmax(self.label,1),tf.argmax(self.pred,1))
        self.accuracy=tf.reduce_mean(tf.cast(correct,tf.float32),name='accuracy')

    def loss_op(self):
        model_loss=-tf.reduce_mean(tf.cast(self.label,tf.float32)*tf.log(self.pred),name='model_loss')
        reg_loss=tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),name='reg_loss')
        self.loss=tf.add(model_loss,reg_loss,name='loss')

    def train_op(self):
        with tf.variable_scope("optimizer"):

            if self.optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            elif self.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
            elif self.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            elif self.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            else :
                print('None optimizer,use SGD!')
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

            tvars = tf.trainable_variables()
            for var in tvars:
                print(var.name, var.shape)
            grads=tf.gradients(self.loss, tvars)
            if self.clip_value is not None:
                grads, _ = tf.clip_by_global_norm(grads,self.clip_value)
            self.optim = optimizer.apply_gradients(
                zip(grads, tvars),
                )

    def biLSTM(self,inputs, hidden_size, length, scope, lstm_drop=0):
        with tf.variable_scope(scope, reuse=False):
            with tf.variable_scope('fcell'):
                fcell = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=tf.orthogonal_initializer())
                if lstm_drop:
                    print('lstm_drop:',lstm_drop)
                    fcell = tf.nn.rnn_cell.DropoutWrapper(fcell, input_keep_prob=lstm_drop)
            with tf.variable_scope('bcell'):
                bcell = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=tf.orthogonal_initializer())
                if lstm_drop:
                    bcell = tf.nn.rnn_cell.DropoutWrapper(bcell, input_keep_prob=lstm_drop)
            (out_fw, out_bw), _ = tf.nn.bidirectional_dynamic_rnn(fcell, bcell, inputs, sequence_length=length,
                                                                  dtype=tf.float32)
            return tf.concat((out_fw, out_bw), 2)

    def linear(self,input,outsize,w_name,b_name=None,activation=None,regularizar=None):
        input_size=input.shape[-1]
        w=tf.get_variable(w_name,[input_size,outsize],regularizer=regularizar)
        out=tf.tensordot(input,w,axes=1)
        if b_name is not None:
            b = tf.get_variable(b_name, shape=[outsize])
            out = out + b
        if activation is not None :
            out=activation(out)
        return out









