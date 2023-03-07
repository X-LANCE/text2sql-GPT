import os
import pickle
from sentence_transformers import SentenceTransformer
from util.example import Example


def encode_dataset(choice, args):
    filename = f'data/{choice}.bin'
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            dataset = pickle.load(file)
        return dataset
    dataset = Example.load_dataset(choice)
    sentence_encoder = SentenceTransformer(os.path.join('plm', args.plm))
    encodings = sentence_encoder.encode(
        [example['question'] for example in dataset],
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_tensor=True,
        device=args.device
    ).cpu().tolist()
    for i, example in enumerate(dataset):
        example['encoding'] = encodings[i]
    with open(filename, 'wb') as file:
        pickle.dump(dataset, file)
    return dataset
