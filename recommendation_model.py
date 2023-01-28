import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell

class Recommendation_Model(object):
    def __init__(self, user_count, item_count, batch_size):
        hidden_size = 128
        memory_window = 10
        
        self.u = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.i = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.y = tf.placeholder(tf.float32, [batch_size,]) # [B]
        self.last = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.hist = tf.placeholder(tf.int32, [batch_size, memory_window]) # [B, T]
        self.lr = tf.placeholder(tf.float64, [])
        
        user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_size])
        item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_size])
        user_b = tf.get_variable("user_b", [user_count], initializer=tf.constant_initializer(0.0))
        item_b = tf.get_variable("item_b", [item_count], initializer=tf.constant_initializer(0.0))

        item_emb = tf.nn.embedding_lookup(item_emb_w, self.i)
        user_emb = tf.nn.embedding_lookup(user_emb_w, self.u)
        item_b = tf.gather(item_b, self.i)
        user_b = tf.gather(user_b, self.u)
        h_emb = tf.nn.embedding_lookup(item_emb_w, self.hist)
        last_emb = tf.nn.embedding_lookup(item_emb_w, self.last)

        # User Preference
        output, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=h_emb, dtype=tf.float32)
        preference, _ = self.seq_attention(output, hidden_size, memory_window)
        preference = tf.nn.dropout(preference, 0.1)

        # Combine User Preferences
        concat = tf.concat([user_emb, preference, item_emb], axis=1)
        concat = tf.layers.batch_normalization(inputs=concat)
        concat = tf.layers.dense(concat, 80, activation=tf.nn.sigmoid, name='f1')
        concat = tf.layers.dense(concat, 40, activation=tf.nn.sigmoid, name='f2')
        concat = tf.layers.dense(concat, 1, activation=None, name='f3')
        concat = tf.reshape(concat, [-1])

        novelty = tf.reduce_mean(h_emb, axis=1)
        novelty = tf.norm(novelty - item_emb ,ord='euclidean', axis=1)

        relevance = 1 - tf.norm(last_emb - item_emb ,ord='euclidean', axis=1)

        #Estmation of user preference by combing different components
        self.logits = item_b + concat + user_b
        self.score = tf.sigmoid(self.logits)

        self.novelty = novelty
        self.relevance = relevance

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step+1)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,labels=self.y))
        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 1)
        self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                self.u: uij[0],
                self.i: uij[1],
                self.hist: uij[2],
                self.y: uij[3],
                self.last: uij[4],
                self.lr: lr,
                })
        return loss

    def compute(self, sess, uij):
        relevance, novelty = sess.run([self.relevance, self.novelty], feed_dict={
                self.u: uij[0],
                self.i: uij[1],
                self.hist: uij[2],
                self.y: uij[3],
                self.last: uij[4]
                })
        return relevance, novelty

    def seq_attention(self, inputs, hidden_size, attention_size):
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape
        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.tile(tf.expand_dims(alphas, -1), [1, 1, hidden_size]), 1, name="attention_embedding")
        return output, alphas