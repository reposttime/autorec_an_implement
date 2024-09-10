import numpy as np

def read_rating(path, num_users, num_items, num_total_ratings, a, b, train_ratio):
  '''
  path: 数据所在目录
  num_users: 用户数
  num_items: 物品数
  num_total_ratings: 总评分数
  a:  1
  b:  0
  train_ratio: 训练集比例
  '''
  fp = open(path + "ratings.dat")

  user_train_set = set()
  user_test_set = set()
  item_train_set = set()
  item_test_set = set()

  R = np.zeros((num_users, num_items)) # 评分矩阵,一行代表一个用户对所有物品的评分
  mask_R = np.zeros((num_users, num_items)) # 标记矩阵, 1表示有评分, 0表示无评分
  C = np.ones((num_users, num_items)) * b # 所有元素都是b的矩阵，tips：这里的C后面似乎没有用到

  train_R = np.zeros((num_users, num_items))
  test_R = np.zeros((num_users, num_items))

  train_mask_R = np.zeros((num_users, num_items))
  test_mask_R = np.zeros((num_users, num_items))

  # 划分训练集和测试集
  random_perm_idx = np.random.permutation(num_total_ratings)
  train_idx = random_perm_idx[0:int(num_total_ratings * train_ratio)]
  test_idx = random_perm_idx[int(num_total_ratings * train_ratio):]

  num_train_ratings = len(train_idx)
  num_test_ratings = len(test_idx)
  
  # 读取评分数据
  lines = fp.readlines()
  for line in lines:
    user, item, rating, _ = line.split("::")
    user_idx = int(user) - 1
    item_idx = int(item) - 1
    R[user_idx, item_idx] = int(rating)
    mask_R[user_idx, item_idx] = 1
    C[user_idx, item_idx] = a # 若有评分,则C中对应位置的值为a

  # 训练集
  for itr in train_idx:
    line = lines[itr]
    user, item, rating, _ = line.split("::")
    user_idx = int(user) - 1
    item_idx = int(item) - 1
    train_R[user_idx, item_idx] = int(rating)
    train_mask_R[user_idx, item_idx] = 1

    user_train_set.add(user_idx)
    item_train_set.add(item_idx)

  # 测试集
  for itr in test_idx:
    line = lines[itr]
    user, item, rating, _ = line.split("::")
    user_idx = int(user) - 1
    item_idx = int(item) - 1
    test_R[user_idx, item_idx] = int(rating)
    test_mask_R[user_idx, item_idx] = 1

    user_test_set.add(user_idx)
    item_test_set.add(item_idx)

  return R, mask_R, C, train_R, train_mask_R, test_R, test_mask_R, num_train_ratings, num_test_ratings, user_train_set, item_train_set, user_test_set, item_test_set
