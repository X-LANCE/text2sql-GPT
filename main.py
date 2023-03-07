import openai
import os
import pickle
import random
import time
from eval.evaluation import isValidSQL
from sentence_transformers import util
from util.arg import main_args
from util.encode import encode_dataset
from util.example import Example
from util.prompt import Prompt


def is_valid_shots(shots, prompt, args):
    return len(prompt.get_prompt(None, None, shots)) < 15000 * len(shots) / (args.cluster_num + args.dynamic_num)


def get_static_shots(dataset, prompt, args):
    if args.zero_shot or args.cluster_num == 0:
        return []
    while 1:
        if args.cluster_method == 'random':
            shots = set()
            while len(shots) < args.cluster_num:
                shots.add(random.randint(0, len(dataset) - 1))
            shots = [dataset[id] for id in shots]
        else:
            shots = []
            for cluster in range(args.cluster_num):
                shots.append(random.choice([example for example in dataset if example['cluster'] == cluster]))
        if is_valid_shots(shots, prompt, args):
            return shots


def get_dynamic_shots(encoding, dataset, prompt, args):
    if args.zero_shot or args.dynamic_num == 0:
        return []
    scores = util.cos_sim(encoding, [example['encoding'] for example in dataset]).squeeze(0).tolist()
    scores = sorted(enumerate(scores), key=lambda x: -x[1])
    shots = []
    for item in scores:
        shots.append(dataset[item[0]])
        if not is_valid_shots(shots, prompt, args):
            shots.pop()
        elif len(shots) == args.dynamic_num:
            break
    return shots


def postprocess(sql):
    return 'SELECT ' + ' '.join(sql.replace('==', '=').replace('<>', '!=').split())


def decode(train_dataset, dev_dataset, args, etype='all'):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    prompt = Prompt(args=args)
    static_shots = get_static_shots(train_dataset, prompt, args)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    pred_filename = os.path.join(args.log_path, 'pred.sql')
    if os.path.exists(pred_filename):
        with open(pred_filename, 'r', encoding='utf-8') as file:
            cached = file.read().count('\n')
        file = open(pred_filename, 'a', encoding='utf-8')
    else:
        cached = 0
        file = open(pred_filename, 'w', encoding='utf-8')
    for i, example in enumerate(dev_dataset):
        print(f'Decoding example {i} ...')
        if i < cached:
            continue
        db_id = example['db_id']
        dynamic_shots = get_dynamic_shots(example['encoding'], train_dataset, prompt, args)
        while 1:
            try:
                response = openai.Completion.create(
                    model='code-davinci-002',
                    prompt=prompt.get_prompt(db_id, example['question'], static_shots + dynamic_shots),
                    max_tokens=150,
                    temperature=0,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=[';', '\n\n', 'Given', 'Translate']
                )
                break
            except:
                print('Retrying ...')
                time.sleep(3)
        sql = postprocess(response['choices'][0]['text'])
        while sql != 'SELECT' and not isValidSQL(sql, f'data/database/{db_id}/{db_id}.sqlite'):
            sql = ' '.join(sql.split()[:-1])
        file.write(sql + '\n')
        file.flush()
    file.close()
    return Example.evaluator.accuracy(pred_filename, dev_dataset, os.path.join(args.log_path, 'dev.txt'), etype=etype)


args = main_args()
random.seed(args.seed)
Example.configuration()
start_time = time.time()
if args.cluster_method == 'random':
    train_dataset = encode_dataset('train', args)
else:
    with open(f'data/train.{args.cluster_method}.{args.cluster_num}.bin', 'rb') as file:
        train_dataset = pickle.load(file)
dev_dataset = encode_dataset('dev', args)
print(f'Dataset size: train -> {len(train_dataset):d}, dev -> {len(dev_dataset):d} ;')
print(f'Load dataset finished, cost {time.time() - start_time:.4f}s ;')
Example.use_database_testsuite()
print('Start evaluating dev dataset on testsuite database ...')
start_time = time.time()
dev_em_acc, dev_ex_acc = decode(train_dataset, dev_dataset, args)
print(f'Evaluation costs {time.time() - start_time:.2f}s, Dev EM/EXT acc: {dev_em_acc:.4f}/{dev_ex_acc:.4f} ;')
