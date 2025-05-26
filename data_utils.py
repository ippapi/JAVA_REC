import numpy as np
import pandas as pd
import ast
from collections import defaultdict

def data_retrieval(path):
    num_users = 0
    num_products = 0
    train = defaultdict(list)
    validation = defaultdict(list)
    test = defaultdict(list)

    def load_train(path, storage):
        nonlocal num_users, num_products
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            user = int(row['user_id'])
            product = int(row['product_id'])
            storage[user].append(product)
            num_users = max(num_users, user)
            num_products = max(num_products, product)

    load_train(path, train)

    for key in list(train.keys()):
        value = train[key]
        if len(value) >= 3:
            val_product = value[-2]
            test_product = value[-1]
            train[key] = value[:-2]
            validation[key].append(val_product)
            test[key].append(test_product)
        else:
            del train[key]

    return [train, validation, test, num_users, num_products]

class Sampler:
    def __init__(self, users_interacts, num_users=1000, num_products=50, batch_size=64, sequence_size=10):
        self.users_interacts = users_interacts
        self.num_users = num_users
        self.num_products = num_products
        self.batch_size = batch_size
        self.sequence_size = sequence_size
        self.user_ids = np.arange(1, self.num_users + 1, dtype=np.int32)
        np.random.seed(1601)
        np.random.shuffle(self.user_ids)
        self.index = 0

    def random_neq(self, l, r, s):
        t = np.random.randint(l, r)
        while t in s:
            t = np.random.randint(l, r)
        return t

    def sample(self, user_id):
        while len(self.users_interacts[user_id]) <= 1:
            user_id = np.random.randint(0, self.num_users)

        seq_product = np.zeros([self.sequence_size], dtype=np.int32)
        pos_product = np.zeros([self.sequence_size], dtype=np.int32)
        neg_product = np.zeros([self.sequence_size], dtype=np.int32)
        next_product = self.users_interacts[user_id][-1]
        next_id = self.sequence_size - 1

        product_set = set(self.users_interacts[user_id])
        for index in reversed(self.users_interacts[user_id][:-1]):
            seq_product[next_id] = index
            pos_product[next_id] = next_product
            if next_product != 0:
                neg_product[next_id] = self.random_neq(1, self.num_products + 1, product_set)
            next_product = index
            next_id -= 1
            if next_id == -1:
                break

        return user_id, seq_product, pos_product, neg_product

    def next_batch(self):
        if self.index + self.batch_size > len(self.user_ids):
            np.random.shuffle(self.user_ids)
            self.index = 0

        batch_user_ids = self.user_ids[self.index:self.index + self.batch_size]
        self.index += self.batch_size

        batch = [self.sample(uid) for uid in batch_user_ids]
        return list(zip(*batch)) 