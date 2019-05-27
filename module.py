import tensorflow as tf

def word_match(x,y,mode='dot'):

        seq_x=tf.shape(x)[1]
        seq_y=tf.shape(y)[1]

        if mode=='dot':
            att_mat=tf.matmul(x,tf.transpose(y,perm=[0,2,1]))

        elif mode=='dot2':
            x_=tf.tile(tf.expand_dims(x,axis=2),[1,1,seq_y,1])
            y_=tf.tile(tf.expand_dims(y,axis=1),[1,seq_x,1,1])
            att_mat=tf.reduce_sum(x_*y_,axis=-1)

        elif mode=='bilinear':
            embed_x=tf.shape(x)[-1]
            embed_y=tf.shape(y)[-1]
            W=tf.get_variable('bilinear_para',shape=[embed_x,embed_y],dtype=tf.float32)
            att_mat=tf.matmul(tf.tensordot(x,W,axes=1),y,transpose_b=True)

        else:
            print('error match!')
            att_mat=None
        return att_mat


def exp_mask(val, mask, name=None):
    """Give very negative number to unmasked elements in val.
    For example, [-3, -2, 10], [True, True, False] -> [-3, -2, -1e9].
    Typically, this effectively masks in exponential space (e.g. softmax)
    Args:
        val: values to be masked
        mask: masking boolean tensor, same shape as tensor
        name: name for output tensor
    Returns:
        Same shape as val, where some elements are very small (exponentially zero)
    """
    VERY_BIG_NUMBER = 1e30
    VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER
    VERY_SMALL_NUMBER = 1e-30
    VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
    if name is None:
        name = "exp_mask"
    return tf.add(val, (1 - tf.cast(mask, 'float')) * VERY_NEGATIVE_NUMBER, name=name)

def mask_softmax(logits, mask=None, scope=None):
    with tf.name_scope(scope or "Softmax"):
        if mask is not None:
            logits = exp_mask(logits, mask)
        flat_out = tf.nn.softmax(logits)
        return flat_out