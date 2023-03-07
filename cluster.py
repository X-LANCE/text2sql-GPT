import pickle
import time
from sklearn.cluster import AgglomerativeClustering, KMeans
from util.arg import cluster_args
from util.encode import encode_dataset


def kmeans(dataset, args):
    model = KMeans(args.cluster)
    model.fit([example['encoding'] for example in dataset])
    for i, example in enumerate(dataset):
        example['cluster'] = model.labels_[i]


def agglomerative(dataset, args):
    model = AgglomerativeClustering(args.cluster)
    model.fit([example['encoding'] for example in dataset])
    for i, example in enumerate(dataset):
        example['cluster'] = model.labels_[i]


args = cluster_args()
start_time = time.time()
dataset = encode_dataset('train', args)
print(f'Dataset size: train -> {len(dataset):d} ;')
print(f'Load dataset finished, cost {time.time() - start_time:.4f}s ;')
print('Start clustering ...')
start_time = time.time()
if args.method == 'kmeans':
    kmeans(dataset, args)
elif args.method == 'agglomerative':
    agglomerative(dataset, args)
else:
    raise ValueError(f'unknown clustering method {args.method}')
print(f'Clustering costs {time.time() - start_time:.2f}s ;')
with open(f'data/train.{args.method}.{args.cluster}.bin', 'wb') as file:
    pickle.dump(dataset, file)
