import json
import os
import pickle
import random
import time
from eval.evaluation import Evaluator, isValidSQL
from eval.process_sql import Schema, get_schema, get_sql
from sentence_transformers import SentenceTransformer
from util.arg import main_args
from util.constant import GPT_CHAT_MODELS, GPT_COMPLETION_MODELS
from util.encode import encode_dataset
from util.example import Example
from util.gpt import get_response
from util.prompt import PromptMaker


def eval_hardness(db_id, sql):
    schema = Schema(get_schema(os.path.join(Example.evaluator.db_dir, db_id, db_id + '.sqlite')))
    return Evaluator().eval_hardness(get_sql(schema, sql))


def postprocess(response, gpt, db_id=None):
    if gpt in GPT_CHAT_MODELS:
        start_idx = response.find('SELECT')
        if start_idx < 0:
            start_idx = response.find('select')
        if start_idx < 0:
            return response
        original_sql = response[start_idx:]
        end_idx = original_sql.find('```')
        if end_idx >= 0:
            original_sql = original_sql[:end_idx]
    elif gpt in GPT_COMPLETION_MODELS:
        original_sql = 'SELECT ' + response
    else:
        raise ValueError(f'unknown GPT model {gpt}')
    original_sql = ' '.join(original_sql.replace('==', '=').replace('<>', '!=').split())
    original_sql = original_sql.replace('INNER JOIN', 'JOIN').replace('inner join', 'join')
    original_sql = original_sql.replace('LEFT JOIN', 'JOIN').replace('left join', 'join')
    if db_id is None:
        return original_sql
    sql = original_sql
    while sql != 'SELECT' and not isValidSQL(sql, os.path.join(Example.evaluator.db_dir, db_id, db_id + '.sqlite')):
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
        with open(pred_filename, 'r', encoding='utf-8') as pred_file:
            cached = pred_file.read().count('\n')
        pred_file = open(pred_filename, 'a', encoding='utf-8')
    else:
        cached = 0
        pred_file = open(pred_filename, 'w', encoding='utf-8')
    if args.two_phase:
        pseudo_filename = os.path.join(args.log_path, 'pseudo.json')
        if os.path.exists(pseudo_filename):
            with open(pseudo_filename, 'r', encoding='utf-8') as pseudo_file:
                pseudo_queries = json.load(pseudo_file)
        else:
            pseudo_queries = {}
    for i, example in enumerate(dev_dataset):
        print(f'Decoding example {i} ...')
        if i < cached:
            continue
        db_id = example['db_id']
        question = example['question']
        query = example['query']
        if args.hard_and_extra and eval_hardness(db_id, query) in ['easy', 'medium']:
            pred_file.write(query.strip('\t ;') + '\n')
            pred_file.flush()
            continue
        if args.zero_shot or args.dynamic_num == 0 or args.encoding == 'question' or args.oracle:
            dynamic_shots = prompt_maker.get_dynamic_shots(example[args.encoding + '_encoding'], train_dataset, args)
        else:
            response = get_response(prompt_maker.get_prompt(args, db_id, question, static_shots), args)
            encoding = sentence_encoder.encode(
                postprocess(response, args.gpt, db_id),
                batch_size=1,
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=args.device
            ).cpu().tolist()
            dynamic_shots = prompt_maker.get_dynamic_shots(encoding, train_dataset, args)
        if args.two_phase:
            prompt_phase_1 = prompt_maker.get_prompt_phase_1(args, question, static_shots + dynamic_shots)
            if str(i) not in pseudo_queries:
                response = get_response(prompt_phase_1, args)
                pseudo_queries[str(i)] = postprocess(response, args.gpt)
                with open(pseudo_filename, 'w', encoding='utf-8') as pseudo_file:
                    json.dump(pseudo_queries, pseudo_file, ensure_ascii=False, indent=4)
            prompt_phase_2 = prompt_maker.get_prompt_phase_2(prompt_phase_1, pseudo_queries[str(i)], db_id)
            response = get_response(prompt_phase_2, args)
        else:
            response = get_response(prompt_maker.get_prompt(args, db_id, question, static_shots + dynamic_shots), args)
        pred_file.write(postprocess(response, args.gpt, db_id) + '\n')
        pred_file.flush()
    pred_file.close()
    return Example.evaluator.accuracy(pred_filename, dev_dataset, os.path.join(args.log_path, 'dev.txt'), etype=etype)


args = main_args()
random.seed(args.seed)
Example.configuration(args.dataset)
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
