import os
import pickle
import random
import time
from eval.evaluation import isValidSQL
from sentence_transformers import SentenceTransformer
from util.arg import main_args
from util.constant import GPT_CHAT_MODELS, GPT_COMPLETION_MODELS
from util.encode import encode_dataset
from util.example import Example
from util.gpt import get_response
from util.prompt import PromptMaker


def postprocess(response, db_id, args):
    if args.gpt in GPT_CHAT_MODELS:
        start_idx = response.find('SELECT')
        if start_idx < 0:
            start_idx = response.find('select')
        if start_idx < 0:
            return response
        original_sql = response[start_idx:]
    elif args.gpt in GPT_COMPLETION_MODELS:
        original_sql = 'SELECT ' + response
    else:
        raise ValueError(f'unknown GPT model {args.gpt}')
    sql = original_sql = ' '.join(original_sql.replace('==', '=').replace('<>', '!=').split())
    while sql != 'SELECT' and not isValidSQL(sql, os.path.join('data', args.dataset, 'database', db_id, db_id + '.sqlite')):
        sql = ' '.join(sql.split()[:-1])
    return original_sql if sql == 'SELECT' else sql


def decode(train_dataset, dev_dataset, args, etype='all'):
    prompt_maker = PromptMaker(args=args)
    static_shots = prompt_maker.get_static_shots(train_dataset, args)
    if args.encoding == 'query' and (not args.oracle):
        sentence_encoder = SentenceTransformer(os.path.join('plm', args.plm))
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
        question = example['question']
        if args.encoding == 'question' or args.oracle:
            dynamic_shots = prompt_maker.get_dynamic_shots(example[args.encoding + '_encoding'], train_dataset, args)
            response = get_response(prompt_maker.get_prompt(args, db_id, question, static_shots + dynamic_shots), args)
        else:
            response = get_response(prompt_maker.get_prompt(args, db_id, question, static_shots), args)
            if args.dynamic_num > 0:
                encoding = sentence_encoder.encode(
                    postprocess(response, db_id, args),
                    batch_size=1,
                    normalize_embeddings=True,
                    convert_to_tensor=True,
                    device=args.device
                ).cpu().tolist()
                dynamic_shots = prompt_maker.get_dynamic_shots(encoding, train_dataset, args)
                response = get_response(prompt_maker.get_prompt(args, db_id, question, static_shots + dynamic_shots), args)
        file.write(postprocess(response, db_id, args) + '\n')
        file.flush()
    file.close()
    return Example.evaluator.accuracy(pred_filename, dev_dataset, os.path.join(args.log_path, 'dev.txt'), etype=etype)


args = main_args()
random.seed(args.seed)
Example.configuration(args)
start_time = time.time()
if args.cluster_method == 'random':
    train_dataset = encode_dataset('train', args)
else:
    with open(os.path.join('data', args.dataset, f'train.{args.cluster_method}.{args.cluster_num}.{args.encoding}.bin'), 'rb') as file:
        train_dataset = pickle.load(file)
dev_dataset = encode_dataset('dev', args)
print(f'Dataset size: train -> {len(train_dataset):d}, dev -> {len(dev_dataset):d} ;')
print(f'Load dataset finished, cost {time.time() - start_time:.4f}s ;')
Example.use_database_testsuite()
print('Start evaluating dev dataset on testsuite database ...')
start_time = time.time()
dev_em_acc, dev_ex_acc = decode(train_dataset, dev_dataset, args)
print(f'Evaluation costs {time.time() - start_time:.2f}s, Dev EM/EXT acc: {dev_em_acc:.4f}/{dev_ex_acc:.4f} ;')
