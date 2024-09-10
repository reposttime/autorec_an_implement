from data_preprocess import *
from AutoRec import AutoRec
import tensorflow as tf
import time
import argparse # 解析命令行参数的库
current_time = time.time()

# 使用解析器添加参数
parser = argparse.ArgumentParser(description='I-AutoRec ')
parser.add_argument('--hidden_neuron', type=int, default=500)
parser.add_argument('--lambda_value', type=float, default=1)

parser.add_argument('--train_epoch', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=100)

parser.add_argument('--optimizer_method', choices=['Adam', 'RMSProp'], default='Adam')
parser.add_argument('--grad_clip', type=bool, default=False)
parser.add_argument('--base_lr', type=float, default=1e-3)
parser.add_argument('--decay_epoch_step', type=int, default=50, help="decay the learning rate for each n epochs")

parser.add_argument('--random_seed', type=int, default=1000)
parser.add_argument('--display_step', type=int, default=1)

args = parser.parse_args() # 解析参数
tf.random.set_seed(args.random_seed)
np.random.seed(args.random_seed)

data_name = 'ml-1m'; num_users = 6040; num_items = 3952; num_total_ratings = 1000209; train_ratio = 0.9
path = "自己实现/data/%s" % data_name + "/"  # 数据路径

result_path = '自己实现/results/' + data_name + '/' + str(args.random_seed) + '_' + str(args.optimizer_method) + '_' + str(args.base_lr) + '_' + str(current_time) + '/' # 结果路径,命名规则为：随机种子_优化器_学习率_时间

R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R, num_train_ratings, num_test_ratings, user_train_set, item_train_set, user_test_set, item_test_set = read_rating(path, num_users, num_items, num_total_ratings, 1, 0, train_ratio) # 数据预处理，得到神经网络所需的数据输入形式

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
   try:
     tf.config.experimental.set_memory_growth(gpus[0], True)
   except RuntimeError as e:
     print(e)

with tf.compat.v1.Session() as sess:
  AutoRec = AutoRec(sess, args, num_users, num_items, R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R, num_train_ratings, num_test_ratings, user_train_set, item_train_set, user_test_set, item_test_set, result_path) # 初始化AutoRec类

  AutoRec.run() # 运行AutoRec类



