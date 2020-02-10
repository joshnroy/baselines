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

def build_reconstructor(inputs):
    out = tf.nn.relu(tf.layers.dense(inputs, 256))
    out = tf.nn.relu(tf.layers.dense(inputs, 1024))
    out = tf.reshape(out, (-1, 4, 4, 64))
    out = tf.nn.relu(tf.layers.conv2d_transpose(out, 64, 3, padding='same', strides=2))
    out = tf.nn.relu(tf.layers.conv2d_transpose(out, 32, 3, padding='same', strides=2))
    out = tf.nn.relu(tf.layers.conv2d_transpose(out, 32, 3, padding='same', strides=2))
    out = tf.nn.sigmoid(tf.layers.conv2d_transpose(out, 3, 3, padding='same', strides=2))

    return out

def build_discriminator(inputs, num_levels):

    layer_num = 0

    def get_layer_num_str():
        nonlocal layer_num
        num_str = str(layer_num)
        layer_num += 1
        return num_str

    def conv_layer(out, depth, strides=(1, 1), kernel_size=3):
        return tf.layers.conv2d(out, depth, kernel_size, padding='same', name='layer_' + get_layer_num_str(), strides=strides)

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value

        out = tf.nn.leaky_relu(inputs)

        out = conv_layer(out, depth)
        out = tf.layers.batch_normalization(out)
        out = tf.nn.leaky_relu(out)
        out = conv_layer(out, depth)
        out = tf.layers.batch_normalization(out)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth, strides=(2, 2))
        out = tf.layers.batch_normalization(out)
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = inputs
    out = tf.nn.leaky_relu(tf.layers.dense(out, 128))
    out = tf.layers.dense(out, num_levels)

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
                nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1, comm=None, microbatch_size=None, disc_coeff=None, num_levels=200):
        self.sess = sess = get_session()

        self.num_levels = num_levels

        if disc_coeff is not None:
            self.disc_coeff = disc_coeff
        else:
            self.disc_coeff = tf.placeholder(tf.float32, [])

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

        with tf.variable_scope('vae'):
            reconstruction = build_reconstructor(train_model.z)

        with tf.variable_scope('discriminator_model', reuse=tf.AUTO_REUSE):
            # CREATE DISCRIMINTATOR MODEL
            discriminator_inputs = train_model.z

            predicted_logits = build_discriminator(discriminator_inputs, num_levels)

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

        self.TRAIN_GEN = tf.placeholder(tf.float32, [])

        # Seed labels for the discriminator
        self.LABELS = LABELS = tf.placeholder(tf.int32, [None])

        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        # VAE-related
        reconstruction_loss = tf.reduce_mean(tf.square(tf.cast(self.train_model.X, tf.float32) - reconstruction * 255.), (1, 2, 3))
        latent_loss = -0.5 * tf.reduce_sum(1. + self.train_model.z_log_std_sq - tf.square(self.train_model.z_mean) - tf.exp(self.train_model.z_log_std_sq), 1)
        vae_loss = tf.reduce_mean(reconstruction_loss + latent_loss)
        

        discriminator_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.LABELS, logits=predicted_logits))
        discriminator_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.LABELS, tf.argmax(predicted_logits, axis=-1, output_type=tf.int32)), tf.float32))

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
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

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # Total loss
        loss = 1. * (pg_loss - entropy * ent_coef + vf_loss * vf_coef)

        pd_loss = tf.reduce_mean(-1. * tf.reduce_sum((1. / float(num_levels) * (tf.nn.log_softmax(predicted_logits, axis=-1))), axis=-1))

        self.update_discriminator_params(comm, discriminator_loss, mpi_rank_weight, LR, max_grad_norm)

        self.update_vae_params(comm, vae_loss, mpi_rank_weight, LR, max_grad_norm=None)

        self.update_policy_params(comm, loss, mpi_rank_weight, LR, max_grad_norm)

        # self.update_all_params(comm, loss + (self.disc_coeff * pd_loss), discriminator_loss, mpi_rank_weight, LR, max_grad_norm)

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'discriminator_loss', 'discriminator_accuracy', 'pd_loss', 'softmax_min', 'softmax_max', 'vae_loss', 'reconstruction_loss', 'latent_loss']
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac, discriminator_loss, discriminator_accuracy, pd_loss, tf.reduce_min(self.predicted_labels), tf.reduce_max(self.predicted_labels), vae_loss, tf.reduce_mean(reconstruction_loss), tf.reduce_mean(latent_loss)]
        if isinstance(self.disc_coeff, tf.Tensor):
            self.loss_names.append("disc_coeff")
            self.stats_list.append(self.disc_coeff)

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101

        self.training_i = 0

    def update_all_params(self, comm, ppo_loss, disc_loss, mpi_rank_weight, LR, max_grad_norm):
        ppo_params = tf.trainable_variables('ppo2_model')
        disc_params = tf.trainable_variables('discriminator_model')

        if comm is not None and comm.Get_size() > 1:
            self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)

        ppo_var_and_grads = self.trainer.compute_gradients(ppo_loss, ppo_params)
        ppo_grads, ppo_var = zip(*ppo_var_and_grads)

        disc_var_and_grads = self.trainer.compute_gradients(disc_loss, disc_params)
        disc_grads, disc_var = zip(*disc_var_and_grads)

        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(ppo_grads + disc_grads, max_grad_norm)

        grads_and_var = list(zip(grads, ppo_var + disc_var))

        self.all_train_op = self.trainer.apply_gradients(grads_and_var)

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

    def update_vae_params(self, comm, loss, mpi_rank_weight, LR, max_grad_norm):
        params = tf.trainable_variables('vae') + tf.trainable_variables('ppo2_model/vae')
        if comm is not None and comm.Get_size() > 1:
            self.vae_trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.vae_trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        grads_and_var = self.vae_trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.vae_grads = grads
        self.vae_var = var
        self.vae_train_op = self.vae_trainer.apply_gradients(grads_and_var)

        return grads

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

        self._disc_train_op = self.disc_trainer.apply_gradients(disc_grads_and_var)

    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, labels, train_disc=None, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        for l in labels:
            if l >= self.num_levels:
                print(l, self.num_levels)
                sys.exit()

        # labels = np.array([np.zeros((8, 8), dtype=np.int64) + l for l in labels])

        td_map = {
            self.train_model.X : obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values,
            self.LABELS : labels,
            self.TRAIN_GEN: 0.,
        }
        if isinstance(self.disc_coeff, tf.Tensor):
            td_map[self.disc_coeff] = (self.training_i / 10000.)

        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        out = self.sess.run(self.stats_list + [self._train_op], td_map)[:-1]
        for _ in range(10):
            self.sess.run([self.vae_train_op], td_map)
        self.sess.run([self._disc_train_op], td_map)
        self.training_i += 1

        # out = self.sess.run(self.stats_list + [self.all_train_op], td_map)[:-1]


        return out

