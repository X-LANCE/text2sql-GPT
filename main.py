import json
import os
import pickle
import random
import time
from eval.evaluation import Evaluator, isValidSQL
from eval.process_sql import Schema, get_schema, get_sql
from sentence_transformers import SentenceTransformer
from util.arg import main_args
from util.constant import GPT_CHAT_MODELS, GPT_COMPLETION_MODELS, SET_OPS, TOT_CLAUSES, TOT_INSTRUCTIONS, TOT_STOPS
from util.encode import encode_dataset
from util.example import Example
from util.gpt import get_response
from util.prompt import PromptMaker


def is_int(token):
    try:
        token = int(token)
        return True
    except:
        return False


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


def postprocess(response, args, db_id=None):
    if args.cot:
        original_sql = response.split('\n')[-1]
    else:
        if args.gpt in GPT_CHAT_MODELS:
            start_idx = response.find('SELECT')
            if start_idx < 0:
                start_idx = response.find('select')
                if start_idx < 0:
                    return response
            original_sql = response[start_idx:]
            end_idx = original_sql.find('```')
            if end_idx >= 0:
                original_sql = original_sql[:end_idx]
        elif args.gpt in GPT_COMPLETION_MODELS:
            original_sql = 'SELECT ' + response
        else:
            raise ValueError(f'unknown GPT model {args.gpt}')
    original_sql = ' '.join(original_sql.replace('==', '=').replace('<>', '!=').split())
    original_sql = original_sql.replace('INNER JOIN', 'JOIN').replace('inner join', 'join')
    original_sql = original_sql.replace('LEFT JOIN', 'JOIN').replace('left join', 'join')
    if db_id is None:
        return original_sql
    sql = original_sql
    while len(sql) > 0 and not isValidSQL(sql, os.path.join(Example.evaluator.db_dir, db_id, db_id + '.sqlite')):
        sql = ' '.join(sql.split()[:-1])
    return sql if len(sql) > 0 else original_sql


def decode(train_dataset, dev_dataset, args, etype='all'):
    prompt_maker = PromptMaker(args=args)
    sentence_encoder = SentenceTransformer(os.path.join('plm', args.plm))
    if args.labeled_shot:
        labeled_shots = load_cached_json_file(os.path.join(args.log_path, 'shot.json'))
    else:
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
    if args.cot:
        cot_filename = os.path.join(args.log_path, 'cot.json')
        cots = load_cached_json_file(cot_filename)
    if args.tot:
        tot_filename = os.path.join(args.log_path, 'tot.json')
        tots = load_cached_json_file(tot_filename)
        eval_shots = load_cached_json_file(os.path.join(args.log_path, 'eval.json'))
    if args.two_phase:
        pseudo_filename = os.path.join(args.log_path, 'pseudo.json')
        pseudo_queries = load_cached_json_file(pseudo_filename)
    for i, example in enumerate(dev_dataset):
        print(f'Decoding example {i} ...')
        if i < cached:
            continue
        db_id = example['db_id']
        query = example['query']
        question = example['question']
        if args.hard_and_extra and eval_hardness(db_id, query) in ['easy', 'medium']:
            pred_file.write(query.strip('\t ;') + '\n')
            pred_file.flush()
            continue
        if args.labeled_shot:
            shots = labeled_shots
        else:
            if args.zero_shot or args.dynamic_num == 0 or args.encoding == 'question' or args.oracle:
                dynamic_shots = prompt_maker.get_dynamic_shots(example[args.encoding + '_encoding'], train_dataset, args)
            else:
                response = get_response(prompt_maker.get_prompt(args, db_id, question, static_shots), args)
                encoding = sentence_encoder.encode(
                    postprocess(response, args, db_id),
                    batch_size=1,
                    normalize_embeddings=True,
                    convert_to_tensor=True,
                    device=args.device
                ).cpu().tolist()
                dynamic_shots = prompt_maker.get_dynamic_shots(encoding, train_dataset, args)
            shots = static_shots + dynamic_shots
        if args.cot:
            cots[str(i)] = {'c_num': args.content + 1}
            response = None
            while response is None:
                cots[str(i)]['c_num'] -= 1
                response = get_response(prompt_maker.get_prompt(args, db_id, question, shots, cots[str(i)]['c_num']), args)
            cots[str(i)]['cot'] = response
            save_cached_json_file(cot_filename, cots)
            pred_file.write(postprocess(response, args, db_id) + '\n')
        elif args.tot:
            static_shots = prompt_maker.get_static_shots(train_dataset, args, 'iue')
            prev_results = [{'db_id': db_id, 'question': question}]
            prev_results[0]['tot_iue'] = get_response(prompt_maker.get_prompt_tot_generate(args, TOT_INSTRUCTIONS, 0, prev_results[0], static_shots + dynamic_shots), args).strip()
            if prev_results[0]['tot_iue'].lower() in SET_OPS:
                response = get_response(prompt_maker.get_prompt(args, db_id, question, shots), args)
                pred_file.write(postprocess(response, args, db_id) + '\n')
                pred_file.flush()
                continue
            tots[str(i)] = {str(step): {} for step in range(1, len(TOT_INSTRUCTIONS))}
            for step in range(1, len(TOT_INSTRUCTIONS)):
                static_shots = prompt_maker.get_static_shots(train_dataset, args, TOT_CLAUSES[step][4:])
                cur_results = []
                for prev_result in prev_results:
                    for _ in range(args.tot_k):
                        cur_results.append(prev_result.copy())
                        response = get_response(prompt_maker.get_prompt_tot_generate(args, TOT_INSTRUCTIONS, step, prev_result, static_shots + dynamic_shots), args, args.tot_t, TOT_STOPS[step]).strip('\t\n .;')
                        end_idx = response.find('\t')
                        if end_idx >= 0:
                            response = response[:end_idx]
                        for clause in ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY']:
                            for prefix in [clause + ': ', clause.lower() + ': ', clause + ' clause: ', clause.lower() + ' clause: ']:
                                if response.startswith(prefix):
                                    response = response[len(prefix):]
                        if step == 2 and (not response.startswith('WHERE ') or 'not needed' in response or 'not required' in response):
                            response = 'The WHERE clause is not needed.'
                        if step == 3 and (not response.startswith('GROUP BY ') or 'not needed' in response or 'not required' in response):
                            response = 'The GROUP BY clause is not needed.'
                        if step == 4 and (not response.startswith('ORDER BY ') or 'not needed' in response or 'not required' in response):
                            response = 'The ORDER BY clause is not needed.'
                        cur_results[-1][TOT_CLAUSES[step]] = response
                beam_size = args.tot_b if step < len(TOT_INSTRUCTIONS) - 1 else 1
                tots[str(i)][str(step)]['tots'] = cur_results
                tots[str(i)][str(step)]['eval'] = get_response(prompt_maker.get_prompt_tot_evaluate(args, cur_results, beam_size, eval_shots), args).strip()
                save_cached_json_file(tot_filename, tots)
                top_ids = [int(k) - 1 for k in tots[str(i)][str(step)]['eval'].split('\n')[-1].replace(',', ' , ').replace('.', ' . ').split() if is_int(k)]
                if len(top_ids) == 0:
                    top_ids = range(min(len(cur_results), beam_size))
                prev_results = [cur_results[k] for k in top_ids[:beam_size]]
            sql = prev_results[0]['tot_select'] + ' ' + prev_results[0]['tot_from']
            if prev_results[0]['tot_where'].startswith('WHERE '):
                sql += ' ' + prev_results[0]['tot_where']
            if prev_results[0]['tot_group_by'].startswith('GROUP BY '):
                sql += ' ' + prev_results[0]['tot_group_by']
            if prev_results[0]['tot_order_by'].startswith('ORDER BY '):
                sql += ' ' + prev_results[0]['tot_order_by']
            pred_file.write(postprocess(sql, args, db_id) + '\n')
        elif args.two_phase:
            prompt_phase_1 = prompt_maker.get_prompt_phase_1(args, question, shots)
            if str(i) not in pseudo_queries:
                if args.oracle:
                    pseudo_queries[str(i)] = example['pseudo_query']
                else:
                    response = get_response(prompt_phase_1, args)
                    pseudo_queries[str(i)] = postprocess(response, args)
                save_cached_json_file(pseudo_filename, pseudo_queries)
            prompt_phase_2 = prompt_maker.get_prompt_phase_2(prompt_phase_1, pseudo_queries[str(i)], db_id)
            response = get_response(prompt_phase_2, args)
            pred_file.write(postprocess(response, args, db_id) + '\n')
        else:
            response = get_response(prompt_maker.get_prompt(args, db_id, question, shots), args)
            pred_file.write(postprocess(response, args, db_id) + '\n')
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
