import os
import pickle
import time
from sklearn.cluster import AgglomerativeClustering, KMeans
from util.arg import cluster_args
from util.encode import encode_dataset


def cluster(dataset, args):
    model = KMeans(args.cluster) if args.method == 'kmeans' else AgglomerativeClustering(args.cluster)
    model.fit([example[args.encoding + '_encoding'] for example in dataset])
    for i, example in enumerate(dataset):
        example['cluster'] = model.labels_[i]


args = cluster_args()
start_time = time.time()
dataset = encode_dataset('train', args)
print(f'Dataset size: train -> {len(dataset):d} ;')
print(f'Load dataset finished, cost {time.time() - start_time:.4f}s ;')
print('Start clustering ...')
start_time = time.time()
cluster(dataset, args)
print(f'Clustering costs {time.time() - start_time:.2f}s ;')
with open(os.path.join('data', args.dataset, f'train.{args.method}.{args.cluster}.{args.encoding}.bin'), 'wb') as file:
    pickle.dump(dataset, file)
