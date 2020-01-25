import tensorflow as tf
import functools
import sys
import numpy as np

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize
from baselines.common.layers import dense

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

def build_discriminator(inputs):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """

    layer_num = 0

    def get_layer_num_str():
        nonlocal layer_num
        num_str = str(layer_num)
        layer_num += 1
        return num_str

    def conv_layer(out, depth):
        return tf.layers.conv2d(out, depth, 3, padding='same', name='layer_' + get_layer_num_str())

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value

        out = tf.nn.leaky_relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.leaky_relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.nn.max_pool(out, ksize=3, strides=2, padding='SAME')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = tf.nn.tanh(inputs[0])
    depths = [32, 32]
    for i in range(len(depths)):
        depth = depths[i]
        out = conv_sequence(out, depth) + tf.nn.tanh(inputs[i+1])

    out = tf.layers.flatten(out)
    out = tf.nn.leaky_relu(out)
    out = tf.layers.dense(out, 200, name='layer_' + get_layer_num_str())

    return out

class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1, comm=None, microbatch_size=None, disc_coeff=1.):
        self.sess = sess = get_session()

        self.disc_coeff = disc_coeff

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)
        with tf.variable_scope('discriminator_model', reuse=tf.AUTO_REUSE):
            # CREATE DISCRIMINTATOR MODEL
            discriminator_inputs = train_model.intermediate_feature

            # predicted_logits = tf.layers.conv2d(discriminator_inputs, filters=32, kernel_size=(4, 4), strides=(2, 2), activation='relu')
            # predicted_logits = tf.nn.leaky_relu(predicted_logits)
            # predicted_logits = tf.layers.conv2d(predicted_logits, filters=32, kernel_size=(4, 4), strides=(2, 2), activation='relu')
            # predicted_logits = tf.nn.leaky_relu(predicted_logits)
            # predicted_logits = tf.layers.conv2d(predicted_logits, filters=64, kernel_size=(4, 4), strides=(2, 2), activation='relu')
            # predicted_logits = tf.nn.leaky_relu(predicted_logits)
            # predicted_logits = tf.layers.flatten(predicted_logits)

            # predicted_logits = tf.nn.leaky_relu(dense(256, 512, "dense1", predicted_logits))
            # # for i in range(2, 2+3):
            # #     predicted_logits = tf.nn.leaky_relu(dense(512, 512, "dense" + str(i), predicted_logits))
            # #     predicted_logits = tf.nn.dropout(predicted_logits, keep_prob=0.8)
            # predicted_logits = dense(512, 200, "dense_out", predicted_logits)

            predicted_logits = build_discriminator(discriminator_inputs)

            self.predicted_labels = tf.nn.softmax(predicted_logits)

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        # Seed labels for the discriminator
        self.LABELS = LABELS = tf.placeholder(tf.int32, [None])

        discriminator_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.LABELS, logits=predicted_logits))
        discriminator_loss_clipped = tf.clip_by_value(discriminator_loss, 0., 6.)
        discriminator_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.LABELS, tf.argmax(predicted_logits, axis=1, output_type=tf.int32)), tf.float32))

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss - discriminator_loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        # ratio = 1.

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        # pg_loss = tf.reduce_mean(pg_losses)
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef# - discriminator_loss * disc_coef

        discriminator_loss *= self.disc_coeff

        self.equal_prob = tf.zeros_like(predicted_logits) + (1. / 200.)
        # pd_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.equal_prob, logits=predicted_logits))
        # pd_loss = tf.reduce_mean(tf.abs(self.predicted_labels - (1. / 200.)))
        pd_loss = -discriminator_loss_clipped
        pd_loss *= self.disc_coeff

        p_grads = self.update_policy_params(comm, loss + pd_loss, mpi_rank_weight, LR, max_grad_norm)
        p_grads = tf.abs(tf.reduce_mean(tf.stack([tf.reduce_mean(g) for g in p_grads if g is not None])))

        # pd_grads = self.update_policy_discriminator_params(comm, pd_loss, mpi_rank_weight, LR, max_grad_norm)
        # pd_grads = tf.abs(tf.reduce_mean(tf.stack([tf.reduce_mean(g) for g in pd_grads if g is not None])))
        self.update_discriminator_params(comm, discriminator_loss, mpi_rank_weight, LR, max_grad_norm)

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'discriminator_loss', 'discriminator_accuracy', 'pd_loss', 'softmax_min', 'softmax_max', 'p_grads']
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac, discriminator_loss, discriminator_accuracy, pd_loss, tf.reduce_min(self.predicted_labels), tf.reduce_max(self.predicted_labels), p_grads]


        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101

        self.training_i = 0

    def update_policy_params(self, comm, loss, mpi_rank_weight, LR, max_grad_norm):
        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model')
        # 2. Build our trainer
        if comm is not None and comm.Get_size() > 1:
            self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)

        return grads

    def update_policy_discriminator_params(self, comm, pd_loss, mpi_rank_weight, LR, max_grad_norm):
        # UPDATE POLICY PARAMETERS USING DISCRIMINTATOR_LOSS
        params = tf.trainable_variables('ppo2_model')
        if comm is not None and comm.Get_size() > 1:
            self.pd_trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            # self.pd_trainer = tf.train.GradientDescentOptimizer(learning_rate=LR)
            self.pd_trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        pd_grads_and_var = self.trainer.compute_gradients(pd_loss, params)
        pd_grads, pd_var = zip(*pd_grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            pd_grads, _pd_grad_norm = tf.clip_by_global_norm(pd_grads, max_grad_norm)
        pd_grads_and_var = list(zip(pd_grads, pd_var))
        self._pd_train_op = self.trainer.apply_gradients(pd_grads_and_var)

        return pd_grads

    def update_discriminator_params(self, comm, discriminator_loss, mpi_rank_weight, LR, max_grad_norm):
        # UPDATE DISCRIMINTATOR PARAMETERS USING DISCRIMINTATOR_LOSS
        # 1. Get the model parameters
        disc_params = tf.trainable_variables('discriminator_model')
        # 2. Build our trainer
        if comm is not None and comm.Get_size() > 1:
            self.disc_trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.disc_trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            # self.disc_trainer = tf.train.GradientDescentOptimizer(learning_rate=LR)
        # 3. Calculate gradients
        disc_grads_and_var = self.disc_trainer.compute_gradients(discriminator_loss, disc_params)
        # disc_grads, disc_var = zip(*disc_grads_and_var)

        # if max_grad_norm is not None:
        #     # Clip the gradients (normalize)
        #     disc_grads, _disc_grad_norm = tf.clip_by_global_norm(disc_grads, max_grad_norm)
        # disc_grads_and_var = list(zip(disc_grads, disc_var))
        self._disc_train_op = self.disc_trainer.apply_gradients(disc_grads_and_var)

    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, labels, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X : obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values,
            self.LABELS : labels
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        out = self.sess.run(
            self.stats_list,
            td_map
        )

        train_frequency = 10

        run_list = [self._train_op]

#         if out[7] < 1.0:
#             run_list += [self._pd_train_op]

        if out[5] > 0.5:
            run_list += [self._disc_train_op]

        self.sess.run(run_list, td_map)

        self.training_i += 1

        return out

