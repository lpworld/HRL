import tensorflow as tf
from tensorflow.nn.rnn_cell import GRUCell

class HRL_Model(object):
    def __init__(self, hidden_size, batch_size, user_embedding, item_embedding, session_embedding, primary_dqn, target_dqn, session_dqn):
        history_length = 10
        gamma = tf.constant([0.99],dtype=tf.float64)
        num_user = len(user_embedding)
        num_item = len(item_embedding)
        
        self.session = tf.placeholder(tf.int64, [batch_size,]) # [B]
        self.user = tf.placeholder(tf.int64, [batch_size,]) # [B]
        self.history = tf.placeholder(tf.int64, [batch_size,history_length]) # [B, H]
        self.rating = tf.placeholder(tf.float64, [batch_size,num_item]) # [B, I]
        self.lr = tf.placeholder(tf.float64, [])

        
        self.user_embedding = tf.constant(user_embedding)
        self.item_embedding = tf.constant(item_embedding)
        self.session_embedding = tf.constant(session_embedding)
        self.primary_dqn = primary_dqn
        self.target_dqn = target_dqn
        self.session_dqn = session_dqn

        user_emb = tf.nn.embedding_lookup(self.user_embedding, self.user)
        session_emb = tf.nn.embedding_lookup(self.session_embedding, self.session)
        h_emb  = tf.nn.embedding_lookup(self.item_embedding, self.history)

        # State Generation
        with tf.variable_scope('gru', reuse=tf.AUTO_REUSE):
            _, state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=h_emb, dtype=tf.float64)

        # Session Policy Generation
        session_policy = tf.layers.dense(tf.concat([user_emb,session_emb],axis=1), hidden_size, activation=tf.nn.sigmoid)

        # Action Generation
        with tf.variable_scope('action_network', reuse=tf.AUTO_REUSE):
            action = tf.layers.batch_normalization(inputs=state)
            action = tf.layers.dense(action, num_item, activation=tf.nn.sigmoid)
            _, self.rec = tf.nn.top_k(action, k=10)
            action = tf.argmax(action, axis=1)
            action_emb = tf.nn.embedding_lookup(self.item_embedding, action)
            action = tf.expand_dims(action, axis=1)
        
        next_history = tf.slice(tf.concat([self.history, action], axis=1), [0, 1], [batch_size, history_length])
        next_h_emb = tf.nn.embedding_lookup(self.item_embedding, next_history)
        reward = tf.gather(params=self.rating, indices=action, axis=1, batch_dims=1)

        # Next State Generation
        with tf.variable_scope('gru', reuse=tf.AUTO_REUSE):
            _, next_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=next_h_emb, dtype=tf.float64)

        # Next Action Generation
        with tf.variable_scope('action_network', reuse=tf.AUTO_REUSE):
            next_action = tf.layers.batch_normalization(inputs=next_state)
            next_action = tf.layers.dense(next_action, num_item, activation=tf.nn.sigmoid)
            next_action = tf.argmax(next_action, axis=1)
            next_action_emb = tf.nn.embedding_lookup(self.item_embedding, next_action)

        # Q-Values Prediction Loss
        with tf.variable_scope('dqn', reuse=tf.AUTO_REUSE):
            q_value = self.primary_dqn.forward(state, action_emb)
            next_q_value = self.target_dqn.forward(next_state, next_action_emb)
            predict_q_value = tf.math.add(reward, tf.multiply(gamma, next_q_value))
            self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=predict_q_value, predictions=q_value))

        # Session Q-Values Prediction Loss
        with tf.variable_scope('session_dqn', reuse=tf.AUTO_REUSE):
            session_q_value = self.session_dqn.forward(state, session_policy)
            next_session_q_value = self.session_dqn.forward(next_state, session_policy)
            predict_session_q_value = tf.math.add(reward, tf.multiply(gamma, next_session_q_value))
            self.meta_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=predict_session_q_value, predictions=session_q_value))

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step+1)
        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss+self.meta_loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 1)
        self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                self.session: uij[0],
                self.user: uij[1],
                self.history: uij[2],
                self.rating: uij[3],
                self.lr: lr
                })
        return loss

    def test(self, sess, uij):
        rec = sess.run(self.rec, feed_dict={
                self.session: uij[0],
                self.user: uij[1],
                self.history: uij[2],
                self.rating: uij[3]
                })
        return rec

class DQN(object):
    def __init__(self, hidden_size, scope_name):
        self.hidden_size = hidden_size
        self.scope_name = scope_name

    def forward(self, state, action):
        with tf.variable_scope(self.scope_name):
            concat = tf.concat([state, action], axis=1)
            q_value = tf.layers.dense(concat, 1, activation=tf.nn.tanh)
            return q_value

class RL_Model(object):
    def __init__(self, hidden_size, batch_size, user_embedding, item_embedding, primary_dqn, target_dqn):
        history_length = 10
        gamma = tf.constant([0.99],dtype=tf.float64)
        num_user = len(user_embedding)
        num_item = len(item_embedding)
        
        self.session = tf.placeholder(tf.int64, [batch_size,]) # [B]
        self.user = tf.placeholder(tf.int64, [batch_size,]) # [B]
        self.history = tf.placeholder(tf.int64, [batch_size,history_length]) # [B, H]
        self.rating = tf.placeholder(tf.float64, [batch_size,num_item]) # [B, I]
        self.lr = tf.placeholder(tf.float64, [])

        
        self.user_embedding = tf.constant(user_embedding)
        self.item_embedding = tf.constant(item_embedding)
        self.primary_dqn = primary_dqn
        self.target_dqn = target_dqn

        user_emb = tf.nn.embedding_lookup(self.user_embedding, self.user)
        h_emb  = tf.nn.embedding_lookup(self.item_embedding, self.history)

        # State Generation
        with tf.variable_scope('gru', reuse=tf.AUTO_REUSE):
            _, state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=h_emb, dtype=tf.float64)

        # Action Generation
        with tf.variable_scope('action_network', reuse=tf.AUTO_REUSE):
            action = tf.layers.batch_normalization(inputs=state)
            action = tf.layers.dense(action, num_item, activation=tf.nn.sigmoid)
            _, self.rec = tf.nn.top_k(action, k=10)
            action = tf.argmax(action, axis=1)
            action_emb = tf.nn.embedding_lookup(self.item_embedding, action)
            action = tf.expand_dims(action, axis=1)
        
        next_history = tf.slice(tf.concat([self.history, action], axis=1), [0, 1], [batch_size, history_length])
        next_h_emb = tf.nn.embedding_lookup(self.item_embedding, next_history)
        reward = tf.gather(params=self.rating, indices=action, axis=1, batch_dims=1)

        # Next State Generation
        with tf.variable_scope('gru', reuse=tf.AUTO_REUSE):
            _, next_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=next_h_emb, dtype=tf.float64)

        # Next Action Generation
        with tf.variable_scope('action_network', reuse=tf.AUTO_REUSE):
            next_action = tf.layers.batch_normalization(inputs=next_state)
            next_action = tf.layers.dense(next_action, num_item, activation=tf.nn.sigmoid)
            next_action = tf.argmax(next_action, axis=1)
            next_action_emb = tf.nn.embedding_lookup(self.item_embedding, next_action)

        # Q-Values Prediction Loss
        with tf.variable_scope('dqn', reuse=tf.AUTO_REUSE):
            q_value = self.primary_dqn.forward(state, action_emb)
            next_q_value = self.target_dqn.forward(next_state, next_action_emb)
            predict_q_value = tf.math.add(reward, tf.multiply(gamma, next_q_value))
            self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=predict_q_value, predictions=q_value))

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step+1)
        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 1)
        self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                self.session: uij[0],
                self.user: uij[1],
                self.history: uij[2],
                self.rating: uij[3],
                self.lr: lr
                })
        return loss

    def test(self, sess, uij):
        rec = sess.run(self.rec, feed_dict={
                self.session: uij[0],
                self.user: uij[1],
                self.history: uij[2],
                self.rating: uij[3]
                })
        return rec