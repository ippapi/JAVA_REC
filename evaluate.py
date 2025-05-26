import sys
import copy
import numpy as np
import random

def evaluate(model, dataset, sequence_size = 10, k = 1):
    [train, validation, test, num_users, num_products] = copy.deepcopy(dataset)

    NDCG = 0.0
    HIT = 0.0
    RECALL = 0.0
    valid_user = 0.0

    users = range(1, num_users + 1)

    for user in users:
        if len(train[user]) < 1 or len(test[user]) < 1:
            continue

        seq_product = np.zeros([sequence_size], dtype=np.int32)
        next_index = sequence_size - 1
        seq_product[next_index] = validation[user][0] if len(validation[user]) > 0 else 0
        next_index -= 1
        for i in reversed(train[user]):
            seq_product[next_index] = i
            next_index -= 1
            if next_index == -1:
                break

        interacted_products = set(train[user])
        interacted_products.add(0)
        predict_products = [test[user][0]]

        all_products = set(range(1, num_products + 1))
        available_products = list(all_products - interacted_products - set(predict_products))
        num_needed = 10 - len(predict_products)
        predict_products += random.sample(available_products, min(num_needed, len(available_products)))

        predictions = -model.predict(*[np.array(l) for l in [[user], [seq_product], predict_products]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()
        valid_user += 1

        if rank < k:
            NDCG += 1 / np.log2(rank + 2)
            HIT += 1
            RECALL += 1

        if valid_user % 10000 == 0:
            print('.', end="")
            sys.stdout.flush()

    if valid_user != 0:
        return {
            "NDCG@k": NDCG / valid_user,
            "Hit@k": HIT / valid_user,
            "Recall@k": RECALL / valid_user
        }
    else:
        return {
            "NDCG@k": 0.0,
            "Hit@k": 0.0,
            "Recall@k": 0.0
        }

def evaluate_validation(model, dataset, sequence_size = 10, k = 1):
    [train, validation, test, num_users, num_products] = copy.deepcopy(dataset)

    NDCG = 0.0
    HIT = 0.0
    RECALL = 0.0
    valid_user = 0.0

    users = range(1, num_users + 1)

    for user in users:
        if len(train[user]) < 1 or len(test[user]) < 1:
            continue

        seq_product = np.zeros([sequence_size], dtype=np.int32)
        next_index = sequence_size - 1
        for i in reversed(train[user]):
            seq_product[next_index] = i
            next_index -= 1
            if next_index == -1:
                break

        interacted_products = set(train[user])
        interacted_products.add(0)
        predict_products = [validation[user][0]]

        all_products = set(range(1, num_products + 1))
        available_products = list(all_products - interacted_products - set(predict_products))
        num_needed = 10 - len(predict_products)
        predict_products += random.sample(available_products, min(num_needed, len(available_products)))

        predictions = -model.predict(*[np.array(l) for l in [[user], [seq_product], predict_products]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()
        valid_user += 1

        if rank < k:
            NDCG += 1 / np.log2(rank + 2)
            HIT += 1
            RECALL += 1

        if valid_user % 10000 == 0:
            print('.', end="")
            sys.stdout.flush()

    if valid_user != 0:
        return {
            "NDCG@k": NDCG / valid_user,
            "Hit@k": HIT / valid_user,
            "Recall@k": RECALL / valid_user
        }
    else:
        return {
            "NDCG@k": 0.0,
            "Hit@k": 0.0,
            "Recall@k": 0.0
        }