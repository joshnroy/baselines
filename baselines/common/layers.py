import tensorflow as tf

def dense(num_in, num_out, name, in_tensor):
    weights = tf.Variable(tf.truncated_normal([num_in, num_out], stddev=0.03), name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_out], stddev=0.01), name=name+'_B')
    out_tensor = tf.matmul(in_tensor, weights) + bias

    return out_tensor
