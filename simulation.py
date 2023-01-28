# backend environment: tensorflow 1.15.0

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

import sys, time
import tensorflow as tf
from simulation_model import HRL_Model, DQN, RL_Model
import scipy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
tf.logging.set_verbosity(tf.logging.ERROR)

class DataInput:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        records = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
        self.i += 1
        session, user, history, rating = [], [], [], []
        for record in records:
            session.append(record[0])
            user.append(record[1])
            history.append(record[2])
            rating.append(record[3])
        return self.i, (session, user, history, rating)

def update_target_graph(primary_network='Primary_DQN', target_network='Target_DQN'):
    # Get the parameters of our Primary Network
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, primary_network)
    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, target_network)
    op_holder = []
    # Update our target_network parameters with DQNNetwork parameters
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def evaluate(sess, model, testset, batch_size, item_embedding):
    record, novelty, diversity = [], [], []
    for _, uij in DataInput(testset, batch_size):
        rec = model.test(sess, uij)
        for batch in range(batch_size):
            last_item = uij[2][batch][-1]
            last_item_embedding = item_embedding[last_item]
            item_embeddings = [item_embedding[x] for x in rec[batch]]
            novelty.append(np.mean([scipy.spatial.distance.cosine(x, last_item_embedding) for x in item_embeddings]))
            diversity.append(scipy.spatial.distance.pdist(item_embeddings, metric='cosine'))
            for index in rec[batch]: #IndexError: list index out of range
                record.append(uij[3][batch][index]) # reward, recommended item
    hit_rate = sum([int(x>0.5) for x in record]) / len(record)
    reward = sum(record) / len(record)
    return hit_rate, reward, np.mean(novelty), np.mean(diversity)

### generate the synthetic data
np.random.seed(625)
num_user = 10000
num_item = 10000
embedding_dim = 50
time_range = 50
session_length = 5
session_num = time_range // session_length

user_embedding = np.random.rand(num_user,embedding_dim) # user feature embeddings
user_embedding = user_embedding / np.linalg.norm(user_embedding, axis=1)[:,None] # normalize user embeddings to the unit ball
item_embedding = np.random.rand(num_item,embedding_dim) # item feature embeddings
item_embedding = item_embedding / np.linalg.norm(item_embedding, axis=1)[:,None] # normalize item embeddings to the unit ball
session_embedding = np.random.rand(session_num,embedding_dim) # session embeddings
rating_user = np.random.normal(0.5,1,num_user) # user intrinsic ratings
user_matrix = np.repeat(np.expand_dims(rating_user,axis=1),num_item,axis=1)
rating_item = np.random.normal(0.5,1,num_item) # item intrinsic ratings
item_matrix = np.repeat(np.expand_dims(rating_item,axis=0),num_user,axis=0)
reward_matrix = np.zeros((num_user,num_item)) # reward matrix
updated_reward = np.zeros((num_user,num_item))
eij = np.random.normal(0,0.1,(num_user,num_item)) # randomized bias terms

trainset, testset = [], []

# Simulate Exploration Intent
user_intent = np.random.normal(0,1,num_user)
intent_matrix = np.repeat(np.expand_dims(user_intent,axis=1),num_item,axis=1)
session_intent = np.random.normal(0,1,session_num)

# Simulate the first 10 item interactions as user history
user_records = np.zeros((num_user,time_range),dtype=np.int64)
reward_matrix = user_matrix + item_matrix + eij + scipy.spatial.distance.cdist(user_embedding, item_embedding, metric='cosine')
scaler = MinMaxScaler()
scaler.fit(reward_matrix)
reward_matrix = scaler.transform(reward_matrix)
for u in range(num_user):
    user_records[u][:10] = np.argpartition(reward_matrix[u],-10)[-10:]

print('History Simulation Completed!')
print(datetime.now())

# Simulate the interaction records in time range 11-40 (as the training data)
for t in range(10,40,1):
    session = t // session_length
    intent = intent_matrix + session_intent[session]
    updated_reward  = reward_matrix + np.multiply(intent, scipy.spatial.distance.cdist(item_embedding, item_embedding[user_records[:,t-1]], metric='cosine'))
    scaler = MinMaxScaler()
    scaler.fit(updated_reward)
    updated_reward = scaler.transform(updated_reward)
    for u in range(num_user):
        user_records[u][t] = np.argmax(updated_reward[u])
        trainset.append((session, u, user_records[u][t-10:t], updated_reward[u]))

print('Training Set Simulation Completed!')
print(datetime.now())

# Simulate the interaction records in time range 41-50 (as the test data)
for t in range(40,50,1):
    session = t // session_length
    intent = intent_matrix + session_intent[session]
    updated_reward  = reward_matrix + np.multiply(intent, scipy.spatial.distance.cdist(item_embedding, item_embedding[user_records[:,-1]], metric='cosine'))
    scaler = MinMaxScaler()
    scaler.fit(updated_reward)
    updated_reward = scaler.transform(updated_reward)
    for u in range(num_user):
        user_records[u][t] = np.argmax(updated_reward[u])
        testset.append((session, u, user_records[u][t-10:t], updated_reward[u]))

print('Test Set Simulation Completed!')
print(datetime.now())

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    lr = 1
    epoch = 20
    batch_size = 50
    sys.stdout.flush()
    primary_dqn = DQN(embedding_dim, 'primary_dqn')
    target_dqn = DQN(embedding_dim, 'target_dqn')
    session_dqn = DQN(embedding_dim, 'session_dqn')
    model = HRL_Model(embedding_dim, batch_size, user_embedding, item_embedding, session_embedding, session_dqn, primary_dqn, target_dqn)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    start_time = time.time()
    for _ in range(epoch):
        for _, uij in DataInput(trainset, batch_size):
            loss = model.train(sess, uij, lr)    
        print('Epoch %d DONE\tCost time: %.2f' % (model.global_epoch_step.eval(), time.time()-start_time))
        # Evaluation Recommendation Performance
        hit_rate, reward, novelty, diversity = evaluate(sess, model, testset, batch_size, item_embedding)
        print('Hit Rate: %.4f\tReward: %.4f\tNovelty: %.4f\tDiversity: %.4f\t' % (hit_rate,reward,novelty,diversity))
        sys.stdout.flush()
        model.global_epoch_step_op.eval()
        update_target_graph('primary_dqn','target_dqn')

print('Evaluation Simulation Completed!')
print(datetime.now())