import tensorflow as tf
import time
import numpy as np
import os
import math

class AutoRec():
  def __init__(self, sess, args, num_users, num_items, R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R, num_train_ratings, num_test_ratings, user_train_set, item_train_set, user_test_set, item_test_set, result_path):
    '''
    sess: 表示当前的会话
    args: 关于神经网络的参数
    num_users,...: 来自于数据准备的信息 
    '''

    self.sess = sess
    self.args = args

    self.num_users = num_users
    self.num_items = num_items
    self.R = R
    self.mask_R = mask_R
    self.C = C # 下面C没有用到
    self.train_R = train_R
    self.train_mask_R = train_mask_R
    self.test_R = test_R
    self.test_mask_R = test_mask_R
    self.num_train_ratings = num_train_ratings
    self.num_test_ratings = num_test_ratings
    self.user_train_set = user_train_set
    self.item_train_set = item_train_set
    self.user_test_set = user_test_set
    self.item_test_set = item_test_set

    self.hidden_neuron = args.hidden_neuron # 隐藏层神经元个数
    self.train_epoch = args.train_epoch # 训练轮数
    self.batch_size = args.batch_size # 批大小
    self.num_batch = int(math.ceil(self.num_users / float(self.batch_size))) # 批次数

    self.base_lr = args.base_lr # 学习率
    self.optimizer_method = args.optimizer_method # 优化方法
    self.display_step = args.display_step # 显示间隔
    self.random_seed = args.random_seed # 随机种子

    self.global_step = tf.Variable(0, trainable=False) # 全局步数
    self.decay_epoch_step = args.decay_epoch_step # 学习率衰减的轮数
    self.decay_step = self.decay_epoch_step * self.num_batch # 学习率衰减的步数
    self.lr = tf.compat.v1.train.exponential_decay(self.base_lr, self.global_step, self.decay_step, 0.96, staircase=True) # 以0.96进行指数衰减
    self.lambda_value = args.lambda_value # 正则化参数

    self.train_cost_list = []
    self.test_cost_list = []
    self.test_rmse_list = []

    self.result_path = result_path
    self.grad_clip = args.grad_clip # 梯度裁剪 当梯度的范数超过预定义的阈值 grad_clip 时，梯度会被缩放到该阈值，从而防止梯度过大

  def run(self):
    self.prepare_model()  # 准备模型
    init = tf.compat.v1.global_variables_initializer() # 初始化所有变量
    self.sess.run(init) # 在会话中执行这个初始化操作
    for epoch_itr in range(self.train_epoch):
      self.train_model(epoch_itr)
      self.test_model(epoch_itr)
    self.make_records()


  def prepare_model(self):
    self.input_R = tf.keras.Input(dtype=tf.float32, shape=(self.num_items,), name='input_R')
    self.input_mask_R = tf.keras.Input(dtype=tf.float32, shape=(self.num_items,), name='input_mask_R') # 接受输入数据的占位符,已修改为tf.keras.Input

    V = tf.Variable(name='V', initial_value=tf.random.truncated_normal(shape=[self.num_items, self.hidden_neuron], mean=0, stddev=0.03), dtype=tf.float32)
    W = tf.Variable(name='W', initial_value=tf.random.truncated_normal(shape=[self.hidden_neuron, self.num_items], mean=0, stddev=0.03), dtype=tf.float32)
    mu = tf.Variable(name='mu', initial_value=tf.zeros(shape=self.hidden_neuron), dtype=tf.float32) 
    b = tf.Variable(name='b', initial_value=tf.zeros(shape=self.num_items), dtype=tf.float32) # 初始化权重变量,已修改为tf.Variable

    # 自动编码器前向传播
    pre_Encoder = tf.matmul(self.input_R, V) + mu
    self.Encoder = tf.nn.sigmoid(pre_Encoder)
    pre_Decoder = tf.matmul(self.Encoder, W) + b
    self.Decoder = tf.identity(pre_Decoder)

    # 误差计算
    pre_rec_cost = tf.multiply((self.input_R - self.Decoder), self.input_mask_R)
    rec_cost = tf.square(self.l2_norm(pre_rec_cost)) # 使用l2范数计算误差
    pre_reg_cost = tf.square(self.l2_norm(W)) + tf.square(self.l2_norm(V))
    reg_cost = self.lambda_value * 0.5 * pre_reg_cost # 正则化项

    self.cost = rec_cost + reg_cost # 总误差

    # 优化
    if self.optimizer_method == 'Adam':
      optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
    elif self.optimizer_method == 'RMSProp':
      optimizer = tf.compat.v1.train.RMSPropOptimizer(self.lr)
    else:
      raise ValueError('Optimizer Key Error')
    
    # 梯度裁剪
    if self.grad_clip:
      gvs = optimizer.compute_gradients(self.cost) # 计算梯度
      capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs] # 梯度裁剪到-5，5之间
      self.optimizer = optimizer.apply_gradients(capped_gvs, global_step=self.global_step) # 应用梯度进行优化
    else:
      self.optimizer = optimizer.minimize(self.cost, global_step=self.global_step)
  
  def train_model(self, itr):
    start_time = time.time()
    random_perm_doc_idx = np.random.permutation(self.num_users) # 随机排列用户

    batch_cost = 0
    for i in range(self.num_batch):
      # 每一批次
      if i == self.num_batch - 1:
        batch_set_idx = random_perm_doc_idx[i * self.batch_size:]
      elif i < self.num_batch - 1:
        batch_set_idx = random_perm_doc_idx[i * self.batch_size : (i + 1) * self.batch_size]

      _, Cost = self.sess.run([self.optimizer, self.cost], feed_dict={self.input_R: self.train_R[batch_set_idx, :], self.input_mask_R : self.train_mask_R[batch_set_idx, :]})

      batch_cost = batch_cost + Cost
    self.train_cost_list.append(batch_cost)

    if (itr+1) % self.display_step == 0:
      print("Training //", "Epoch %d //" % (itr), "Total cost = {:.2f}".format(batch_cost), "Elapsed time : %d sec" % (time.time() - start_time))
  
  def test_model(self, itr):
    start_time = time.time()
    Cost, Decoder = self.sess.run([self.cost, self.Decoder], feed_dict={self.input_R: self.test_R, self.input_mask_R: self.test_mask_R})

    self.test_cost_list.append(Cost)

    if (itr+1) % self.display_step == 0:
      Estimated_R = Decoder.clip(min=1, max=5) # 解码结果限制在1-5之间
      unseen_user_test_list = list(self.user_test_set - self.user_train_set)
      unseen_item_test_list = list(self.item_test_set - self.item_train_set)

      for user in unseen_user_test_list:
        for item in unseen_item_test_list:
          if self.test_mask_R[user, item] == 1: # 如果测试集中有评分
            Estimated_R[user, item] = 3

      pre_numerator = np.multiply((Estimated_R - self.test_R), self.test_mask_R)
      numerator = np.sum(np.square(pre_numerator))
      denominator = self.num_test_ratings
      RMSE = np.sqrt(numerator / float(denominator))

      self.test_rmse_list.append(RMSE)

      print("Testing //", "Epoch %d //" % (itr), "Total cost = {:.2f}".format(Cost), "RMSE = {:.5f}".format(RMSE), "Elapsed time : %d sec" % (time.time() - start_time))
      print("=" * 100)
  
  def make_records(self):
    if not os.path.exists(self.result_path):
      os.makedirs(self.result_path)

    basic_info = self.result_path + "basic_info.txt"
    train_record = self.result_path + "train_record.txt"
    test_record = self.result_path + "test_record.txt"

    with open(train_record, "w") as f:
      f.write(str("Cost:"))
      f.write('\t')
      for itr in range(len(self.train_cost_list)):
        f.write(str(self.train_cost_list[itr]))
        f.write('\t')
      f.write('\n')

    with open(test_record, 'w') as g:
      g.write(str("Cost:"))
      g.write('\t')
      for itr in range(len(self.test_cost_list)):
        g.write(str(self.test_cost_list[itr]))
        g.write('\t')
      g.write('\n')

      g.write(str("RMSE:"))
      for itr in range(len(self.test_rmse_list)):
        g.write(str(self.test_rmse_list[itr]))
        g.write('\t')
      g.write('\n')

    with open(basic_info, 'w') as h:
      h.write(str(self.args))
  
  def l2_norm(self, tensor):
    return tf.sqrt(tf.reduce_sum(tf.square(tensor)))