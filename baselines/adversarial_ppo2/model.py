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

def build_discriminator(inputs, num_levels):
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

    out = tf.nn.tanh(inputs)

    out = tf.nn.leaky_relu(tf.layers.dense(out, 512, name='impala_layer_' + get_layer_num_str()), name='impala_layer_' + get_layer_num_str())
    out = tf.nn.leaky_relu(tf.layers.dense(out, 512, name='impala_layer_' + get_layer_num_str()), name='impala_layer_' + get_layer_num_str())
    out = tf.layers.dense(out, 1, name='impala_layer_' + get_layer_num_str())

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

        if self.disc_coeff > 0.:
            with tf.variable_scope('discriminator_model', reuse=tf.AUTO_REUSE):
                # CREATE DISCRIMINTATOR MODEL
                discriminator_inputs = tf.concat([train_model.train_intermediate_feature, train_model.test_intermediate_feature], 0)

                predicted_logits = build_discriminator(discriminator_inputs, num_levels)

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])

        self.TRAIN_GEN = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        if self.disc_coeff > 0.:
            self.real_labels_loss = tf.reduce_mean(predicted_logits[:nbatch_train, 0])
            self.fake_labels_loss = tf.reduce_mean(predicted_logits[nbatch_train:, 0])
            discriminator_loss = -self.real_labels_loss + self.fake_labels_loss

            # self.real_mean_loss = tf.reduce_mean(train_model.train_intermediate_feature, 0)
            # self.fake_mean_loss = tf.reduce_mean(train_model.test_intermediate_feature, 0)
            # self.real_mean_loss = train_model.train_intermediate_feature
            # self.fake_mean_loss = train_model.test_intermediate_feature
            # mmd_loss = tf.reduce_mean((self.real_mean_loss - self.fake_mean_loss)**2.)
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
        rl_loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef# - discriminator_loss * disc_coef

        if self.disc_coeff > 0.:
            pd_loss = self.real_labels_loss - self.fake_labels_loss
            # pd_loss = -self.fake_labels_loss
            # pd_loss = mmd_loss

        loss = rl_loss
        if self.disc_coeff > 0.:
            loss +=  self.TRAIN_GEN * self.disc_coeff * pd_loss

        if self.disc_coeff > 0.:
            self.update_discriminator_params(comm, discriminator_loss, mpi_rank_weight, LR, max_grad_norm)

        self.update_policy_params(comm, loss, mpi_rank_weight, LR, max_grad_norm)

        state_variance = tf.reduce_mean(tf.math.reduce_std(train_model.train_intermediate_feature, axis=0))

        stats_dict = {
            'policy_loss': pg_loss,
            'value_loss': vf_loss,
            'policy_entropy': entropy,
            'approxkl': approxkl,
            'clipfrac': clipfrac,
            'state_variance': state_variance
        }

        if self.disc_coeff > 0.:
            stats_dict.update({
                'discriminator_loss': discriminator_loss,
                'pd_loss': pd_loss,
                'critic_min': tf.reduce_min(predicted_logits),
                'critic_max': tf.reduce_max(predicted_logits),
                'real_labels_loss': self.real_labels_loss,
                'fake_labels_loss': self.fake_labels_loss
            })

        self.loss_names = []
        self.stats_list = []
        for k in stats_dict.keys():
            self.loss_names.append(k)
            self.stats_list.append(stats_dict[k])

        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.step_evaluate = act_model.step_evaluate
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101

        self.training_i = 0

        self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in tf.trainable_variables('discriminator_model')]

    def update_policy_params(self, comm, loss, mpi_rank_weight, LR, max_grad_norm):
        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model')
        # params = [p for p in params if "cnn" not in p.name]
        # 2. Build our trainer
        if comm is not None and comm.Get_size() > 1:
            self.policy_trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            # self.policy_trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            self.policy_trainer = tf.train.RMSPropOptimizer(learning_rate=LR)
        # 3. Calculate the gradients
        grads_and_var = self.policy_trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self.policy_train_op = self.policy_trainer.apply_gradients(grads_and_var)

        return grads

    def update_discriminator_params(self, comm, discriminator_loss, mpi_rank_weight, LR, max_grad_norm):
        # UPDATE DISCRIMINTATOR PARAMETERS USING DISCRIMINTATOR_LOSS
        # 1. Get the model parameters
        disc_params = tf.trainable_variables('discriminator_model')
        # 2. Build our trainer
        if comm is not None and comm.Get_size() > 1:
            self.disc_trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            # self.disc_trainer = tf.train.AdamOptimizer(learning_rate=LR, beta1=0.5, beta2=0.999)
            self.disc_trainer = tf.train.RMSPropOptimizer(learning_rate=LR)
            # self.disc_trainer = tf.train.GradientDescentOptimizer(learning_rate=LR)
        # 3. Calculate gradients
        disc_grads_and_var = self.disc_trainer.compute_gradients(discriminator_loss, disc_params)
        disc_grads, disc_var = zip(*disc_grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            disc_grads, _disc_grad_norm = tf.clip_by_global_norm(disc_grads, max_grad_norm)
        disc_grads_and_var = list(zip(disc_grads, disc_var))
        self.discriminator_train_op = self.disc_trainer.apply_gradients(disc_grads_and_var)

    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, labels, eval_obs, eval_labels, train_disc=None, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        for l in labels:
            if l >= self.num_levels:
                print(l, self.num_levels)
                sys.exit()

        td_map = {
            self.train_model.train_X : obs,
            self.train_model.test_X : eval_obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values,
            self.TRAIN_GEN : self.training_i % 5 == 0,
        }

        if self.disc_coeff == 0.0:
            td_map[self.TRAIN_GEN] = 0
            out = self.sess.run(self.stats_list + [self.policy_train_op], td_map)[:len(self.stats_list)]
        else:
            out = self.sess.run(self.stats_list + [self.policy_train_op, self.discriminator_train_op], td_map)[:len(self.stats_list)]
            self.sess.run([self.clip_D])
        self.training_i += 1

        return out

