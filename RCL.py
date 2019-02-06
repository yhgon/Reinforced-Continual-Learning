# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 19:27:34 2018

@author: Jason
"""
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
from evaluate import evaluate
from policy_gradient import Controller
import argparse
import datetime
import time
import pickle

class RCL:
    def __init__(self,args):
        self.args = args
        self.num_tasks = args.n_tasks
        self.epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.data_path = args.data_path
        self.max_trials = args.max_trials
        self.penalty = args.penalty
        self.task_list = self.create_mnist_task()
        self.evaluates = evaluate(task_list=self.task_list, args = args)
        self.train()

    def create_session(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        return sess

    def create_mnist_task(self):
        data = pickle.load(open(self.data_path, "rb"))
        return data

    def train(self):
        self.best_params={}
        self.result_process = []
        for task_id in range(0,self.num_tasks):
            self.best_params[task_id] = [0,0]
            if task_id == 0:
                print("DEBUG : taks{}/{} IF".format(task_id, self.num_tasks) )
                with tf.Graph().as_default() as g:
                    with tf.name_scope("before"):
                        inputs = tf.placeholder(shape=(None, 784), dtype=tf.float32)
                        y = tf.placeholder(shape=(None, 10), dtype=tf.float32)
                        w1 = tf.Variable(tf.truncated_normal(shape=(784,312), stddev=0.01))
                        b1 = tf.Variable(tf.constant(0.1, shape=(312,)))
                        w2 = tf.Variable(tf.truncated_normal(shape=(312,128), stddev=0.01))
                        b2 = tf.Variable(tf.constant(0.1, shape=(128,)))
                        w3 = tf.Variable(tf.truncated_normal(shape=(128,10), stddev=0.01))
                        b3 = tf.Variable(tf.constant(0.1, shape=(10,)))
                        output1 = tf.nn.relu(tf.nn.xw_plus_b(inputs,w1,b1,name="output1"))
                        output2 = tf.nn.relu(tf.nn.xw_plus_b(output1,w2,b2,name="output2"))
                        output3 = tf.nn.xw_plus_b(output2,w3,b3,name="output3")
                        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output3)) + \
                               0.0001*(tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3))
                        if self.args.optimizer=="adam":
                            optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
                        elif self.args.optimizer=="rmsprop":
                            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
                        elif self.args.optimizer=="sgd":
                            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
                        else:
                            raise Exception("please choose one optimizer")
                        train_step = optimizer.minimize(loss)
                        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,axis=1),tf.argmax(output3,axis=1)),tf.float32))
                        sess = self.create_session()
                        sess.run(tf.global_variables_initializer())
                        l = len(self.task_list[0][1])
                        for epoch in range(self.epochs):
                            print("task {}/{}  epoch {} run for {} IF ".format(task_id, self.num_tasks, epoch, l  ) )                            
                            flag = 0
                            for _ in range(l//self.batch_size+1):                                
                                batch_xs, batch_ys = (self.task_list[task_id][0][flag:flag+self.batch_size],self.task_list[task_id][1][flag:flag+self.batch_size])
                                flag += self.batch_size
                                sess.run(train_step,feed_dict={inputs:batch_xs, y:batch_ys})
                                print("task {}/{}  epoch {}  train_step {}/{}".format(task_id, self.num_tasks, epoch, flag, l) )
                        accuracy_test = sess.run(accuracy, feed_dict={inputs:self.task_list[task_id][4], y:self.task_list[task_id][5]})
                        print("task {}/{} test accuracy: {} IF".format(task_id, self.num_tasks, accuracy_test) )
                        self.vars = sess.run([w1,b1,w2,b2,w3,b3])
                    self.best_params[task_id] = [accuracy_test,self.vars]
            else:
                print("DEBUG : taks{}/{} ELSE".format(task_id, self.num_tasks) )
                controller = Controller(self.args)
                print("DEBUG : taks{}/{} ELSE finish Controller ".format(task_id, self.num_tasks) )
                results = []
                print("DEBUG : taks{}/{} ELSE result Controller ".format(task_id, self.num_tasks) )
                best_reward = 0
                print("DEBUG : taks{}/{} ELSE start trial loop ".format(task_id, self.num_tasks) )
                for trial in range(self.max_trials):
                    actions = controller.get_actions()
                    print("task {}/{} trial {}/{} *********actions  for {} ELSE ".format(task_id, self.num_tasks, trial, self.max_trials, actions) )
                    accuracy_val, accuracy_test = self.evaluates.evaluate_action(var_list = self.vars, 
                             actions=actions, task_id = task_id)

                    results.append(accuracy_val)
                    print("task {}/{}  trial {}/{}, test accuracy: {} ELSE".format(task_id, self.num_tasks, trial, self.max_trials, accuracy_test) )
                    reward = accuracy_val - self.penalty*sum(actions)
                    print("                              reward: {} ELSE".format(reward) )
                    if reward > best_reward:
                        best_reward = reward
                        self.best_params[task_id] = (accuracy_test, self.evaluates.var_list)
                        print("DEBUG : taks{}/{} trial {}/{} ELSE done best_reward ".format(task_id, self.num_tasks, trial, self.max_trials) )
                    controller.train_controller(reward)
                    print("DEBUG : taks{}/{} trial {}/{} ELSE end trial internal loop for train_control ".format(task_id, self.num_tasks,trial, self.max_trials) )
                    
                print("DEBUG : taks{}/{} ELSE end trial loop ".format(task_id, self.num_tasks) )
                controller.close_session()
                print("DEBUG : taks{}/{} ELSE end controller session ".format(task_id, self.num_tasks) )
                self.result_process.append(results)
                print("DEBUG : taks{}/{} ELSE end result append ".format(task_id, self.num_tasks) )
                self.vars = self.best_params[task_id][1]
                print("DEBUG : taks{}/{} ELSE end self vars ".format(task_id, self.num_tasks) )
                print("DEBUG : taks{}/{} ELSE end loop ".format(task_id, self.num_tasks) )
        
if __name__ == "__main__":
    print("DEBUG : RCL.py step1. start...")
    parser = argparse.ArgumentParser(description='Reinforced Continual learning')

    # model parameters
    parser.add_argument('--n_tasks', type=int, default=10,
                        help='number of tasks')
    parser.add_argument('--n_hiddens', type=str, default='312,218',
                        help='number of hidden neurons at each layer')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of hidden layers')

    # optimizer parameters
    parser.add_argument('--n_epochs', type=int, default=15,
                        help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='SGD learning rate')
    parser.add_argument('--max_trials', type=int, default=50,
                        help='max_trials')

    # experiment parameters
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')

    parser.add_argument('--save_path', type=str, default='/mnt/dataset/lsun/Reinforced-Continual-Learning/results/',
                        help='save models at the end of training')

    # data parameters
    parser.add_argument('--data_path', default='/mnt/dataset/lsun/Reinforced-Continual-Learning/mnist_permutations.pkl',
                        help='path where data is located')
    parser.add_argument('--state_space', type=int, default=30, help="the state space for search") 
    parser.add_argument('--actions_num', type=int, default=2, help="how many actions to dscide")
    parser.add_argument('--hidden_size', type=int, default=100, help="the hidden size of RNN")
    parser.add_argument('--num_layers', type=int, default=2, help="the layer of a RNN cell")
    parser.add_argument('--cuda', type=bool, default=True, help="use GPU or not")
    parser.add_argument('--bendmark', type=str, default='critic', help="the type of bendmark")
    parser.add_argument('--penalty', type=float, default=0.0001, help="the type of bendmark")#0.0001
    parser.add_argument('--optimizer', type=str, default="adam", help="the type of optimizer")#
    parser.add_argument('--method', type=str, default='policy', help="method for generate actions")

    args = parser.parse_args()
    start = time.time()
    print("DEBUG : RCL.py start RCL train")
    jason = RCL(args)
    print("DEBUG : RCL.py end RCL train")
    end = time.time()
    params = jason.best_params
    print("DEBUG : RCL.py step2. argparse finished...")
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    print("DEBUG : RCL.py step3. mkdir finished...")
    fname = "RCL_FC_" + args.data_path.split('/')[-1] + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    fname += '_' + str(args.lr) + str("_") + str(args.n_epochs) + '_' + str(args.max_trials) + '_' + str(args.batch_size) + \
             '_' + args.bendmark + '_' + str(args.penalty) + '_' + args.optimizer + '_' + str(args.state_space) + '_' + \
             str(end-start) + '_' + args.method
    fname = os.path.join(args.save_path, fname)
    print("DEBUG : RCL.py step4. fsave finished...")
    f = open(fname + '.txt', 'w')
    accuracy = []
    for index,value in params.items():
        print([_.shape for _ in value[1]], file=f)
        accuracy.append(value[0])
    print(accuracy,file=f)
    f.close()
    print(fname)
    print("DEBUG : RCL.py step5. fclose for txt  finished...")
    name = fname + '.pkl'
    f = open(name, 'wb')
    pickle.dump(jason.result_process, f)
    f.close()
    print("DEBUG : RCL.py step6. fclose for pkl finished...")
