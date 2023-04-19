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


def load_cached_json_file(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            content = json.load(file)
    else:
        content = {}
    return content


def save_cached_json_file(filename, content):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(content, file, ensure_ascii=False, indent=4)


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
    def decode_one_example(question, encoding=None):
        if args.zero_shot or args.dynamic_num == 0 or args.encoding == 'question' or args.oracle:
            if encoding is None:
                encoding = example[args.encoding + '_encoding']
            dynamic_shots = prompt_maker.get_dynamic_shots(encoding, train_dataset, args)
        else:
            response = get_response(prompt_maker.get_prompt(args, db_id, question, static_shots), True, args)
            encoding = sentence_encoder.encode(
                postprocess(response, args.gpt, db_id),
                batch_size=1,
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=args.device
            ).cpu().tolist()
            dynamic_shots = prompt_maker.get_dynamic_shots(encoding, train_dataset, args)
        if args.subproblem > 1 and len(subqa[str(i)]['a']) == args.subproblem:
            response = get_response(prompt_maker.get_prompt(args, db_id, question, static_shots + dynamic_shots, subqa[str(i)]), True, args)
        elif args.two_phase:
            prompt_phase_1 = prompt_maker.get_prompt_phase_1(args, question, static_shots + dynamic_shots)
            if str(i) not in pseudo_queries:
                if args.oracle:
                    pseudo_queries[str(i)] = example['pseudo_query']
                else:
                    response = get_response(prompt_phase_1, True, args)
                    pseudo_queries[str(i)] = postprocess(response, args.gpt)
                save_cached_json_file(pseudo_filename, pseudo_queries)
            prompt_phase_2 = prompt_maker.get_prompt_phase_2(prompt_phase_1, pseudo_queries[str(i)], db_id)
            response = get_response(prompt_phase_2, True, args)
        else:
            response = get_response(prompt_maker.get_prompt(args, db_id, question, static_shots + dynamic_shots), True, args)
        return postprocess(response, args.gpt, db_id)

    prompt_maker = PromptMaker(args=args)
    sentence_encoder = SentenceTransformer(os.path.join('plm', args.plm))
    static_shots = prompt_maker.get_static_shots(train_dataset, args)
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
    if args.subproblem > 1:
        subqa_filename = os.path.join(args.log_path, 'subqa.json')
        subqa = load_cached_json_file(subqa_filename)
    if args.two_phase:
        pseudo_filename = os.path.join(args.log_path, 'pseudo.json')
        pseudo_queries = load_cached_json_file(pseudo_filename)
    for i, example in enumerate(dev_dataset):
        print(f'Decoding example {i} ...')
        if i < cached:
            continue
        db_id = example['db_id']
        query = example['query']
        if args.hard_and_extra and eval_hardness(db_id, query) in ['easy', 'medium']:
            pred_file.write(query.strip('\t ;') + '\n')
            pred_file.flush()
            continue
        if args.subproblem == 1:
            pred_file.write(decode_one_example(example['question']) + '\n')
            pred_file.flush()
            continue
        if str(i) not in subqa:
            subqa[str(i)] = {'q_raw': [], 'q': [], 'a': []}
            response = get_response(prompt_maker.get_prompt_split_problem(args, example['question']), False, args)
            if args.gpt in GPT_COMPLETION_MODELS:
                response = '1. ' + response
            for j in range(1, args.subproblem + 1):
                start_idx = response.find(str(j) + '. ')
                if start_idx < 0:
                    start_idx = response.find(str(j) + ': ')
                if start_idx >= 0:
                    response = response[start_idx + 3:]
                end_idx = response.find('\n')
                if end_idx < 0 and j < args.subproblem:
                    end_idx = response.find(str(j + 1) + '. ')
                    if end_idx < 0:
                        end_idx = response.find(str(j + 1) + ': ')
                if end_idx < 0:
                    end_idx = len(response)
                subqa[str(i)]['q_raw'].append(response[:end_idx].strip())
                response = response[end_idx:]
            save_cached_json_file(subqa_filename, subqa)
        if args.subproblem == 2 and len(subqa[str(i)]['q']) == 0:
            response = get_response(prompt_maker.get_prompt_remove_context_dependency(args, subqa[str(i)]['q_raw']), False, args)
            start_idx = response.find('2. ')
            if start_idx >= 0:
                response = response[start_idx + 3:]
            subqa[str(i)]['q'].append(subqa[str(i)]['q_raw'][0])
            subqa[str(i)]['q'].append(response.strip())
            save_cached_json_file(subqa_filename, subqa)
        for j in range(len(subqa[str(i)]['a']), args.subproblem):
            encoding = sentence_encoder.encode(
                subqa[str(i)]['q'][j],
                batch_size=1,
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=args.device
            ).cpu().tolist()
            subqa[str(i)]['a'].append(decode_one_example(subqa[str(i)]['q'][j], encoding))
            save_cached_json_file(subqa_filename, subqa)
        pred_file.write(decode_one_example(example['question']) + '\n')
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
