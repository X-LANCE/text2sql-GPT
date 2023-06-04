import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
import sqlite3
from sentence_transformers import util
from util.constant import GPT_CHAT_MODELS, GPT_COMPLETION_MODELS, SET_OPS, TOT_CLAUSES


class PromptMaker:
    def __init__(self, args):
        with open(os.path.join('data', args.dataset, 'tables.json'), 'r', encoding='utf-8') as file:
            dbs = json.load(file)
        self.db_prompts = {}
        for db in dbs:
            db_id = db['db_id']
            tabs = db['table_names_original']
            cols = db['column_names_original']
            self.db_prompts[db_id] = [''] * (args.content + 1)
            for c_num in range(args.content + 1):
                for i in range(len(tabs)):
                    if args.api_doc:
                        self.db_prompts[db_id][c_num] += f"# {tabs[i]}({', '.join(col[1] for col in cols if col[0] == i)})\n"
                    else:
                        self.db_prompts[db_id][c_num] += f'create table {tabs[i]} (\n'
                        for j in range(len(cols)):
                            if cols[j][0] == i:
                                self.db_prompts[db_id][c_num] += f"    {cols[j][1]} {db['column_types'][j]}"
                                if args.pf == 'eoc':
                                    if j in db['primary_keys']:
                                        self.db_prompts[db_id][c_num] += ' primary key'
                                    for fk in db['foreign_keys']:
                                        if fk[0] == j:
                                            self.db_prompts[db_id][c_num] += f' references {tabs[cols[fk[1]][0]]}({cols[fk[1]][1]})'
                                self.db_prompts[db_id][c_num] += ',\n'
                        if args.pf == 'eot':
                            pks = [cols[pk][1] for pk in db['primary_keys'] if cols[pk][0] == i]
                            if len(pks) > 0:
                                self.db_prompts[db_id][c_num] += f"    primary key ({', '.join(pks)}),\n"
                            for fk in db['foreign_keys']:
                                if cols[fk[0]][0] == i:
                                    self.db_prompts[db_id][c_num] += f'    foreign key ({cols[fk[0]][1]}) references {tabs[cols[fk[1]][0]]}({cols[fk[1]][1]}),\n'
                        self.db_prompts[db_id][c_num] = self.db_prompts[db_id][c_num][:-2] + '\n)\n'
                    if c_num > 0:
                        conn = sqlite3.connect(os.path.join('data', args.dataset, 'database', db_id, db_id + '.sqlite'))
                        conn.row_factory = dict_factory
                        cursor = conn.cursor()
                        db_contents = cursor.execute(f'SELECT * FROM {tabs[i]} LIMIT {c_num}').fetchall()
                        self.db_prompts[db_id][c_num] += '/*\n'
                        self.db_prompts[db_id][c_num] += f"{len(db_contents)} example row{'s' if len(db_contents) > 1 else ''} from table {tabs[i]}:\n"
                        self.db_prompts[db_id][c_num] += '\t'.join([col[1] for col in cols if col[0] == i]) + '\n'
                        for record in db_contents:
                            self.db_prompts[db_id][c_num] += '\t'.join([str(record[col[1]]) for col in cols if col[0] == i]) + '\n'
                        self.db_prompts[db_id][c_num] += '*/\n'
                self.db_prompts[db_id][c_num] = self.db_prompts[db_id][c_num][:-1]

    def get_prompt(self, args, db_id=None, question=None, shots=[], c_num=-1):
        if c_num < 0:
            c_num = args.content
        if args.gpt in GPT_CHAT_MODELS:
            prompt = [{'role': 'system', 'content': 'Given the database schema, you need to translate the question into the SQL query.'}]
            for shot in shots:
                prompt.append({'role': 'user', 'content': f"Database schema:\n{self.db_prompts[shot['db_id']][c_num]}\nQuestion: {shot['question']}"})
                if args.cot:
                    prompt.append({'role': 'assistant', 'content': shot['cot']})
                else:
                    prompt.append({'role': 'assistant', 'content': shot['query']})
            if db_id and question:
                prompt.append({'role': 'user', 'content': f'Database schema:\n{self.db_prompts[db_id][c_num]}\nQuestion: {question}'})
        elif args.gpt in GPT_COMPLETION_MODELS:
            prompt = ''
            for shot in shots:
                prompt += 'Given the database schema:\n'
                prompt += self.db_prompts[shot['db_id']][c_num] + '\n'
                prompt += 'Translate the natural utterance into the SQL query: ' + shot['question'] + '\n'
                if args.cot:
                    prompt += shot['cot'] + '\n'
                else:
                    prompt += shot['query'] + '\n'
            if db_id and question:
                prompt += 'Given the database schema:\n'
                prompt += self.db_prompts[db_id][c_num] + '\n'
                prompt += 'Translate the natural utterance into the SQL query: ' + question + '\n'
                if args.cot:
                    prompt += "Let's think step by step."
                else:
                    prompt += 'SELECT'
        else:
            raise ValueError(f'unknown GPT model {args.gpt}')
        return prompt

    def get_prompt_tot_generate(self, args, instructions, step, prev_result, shots=[]):
        def add_item(item):
            content = f"Database schema:\n{self.db_prompts[item['db_id']][args.content]}\nQuestion: {item['question']}"
            if step > 1:
                content += '\nSELECT clause: ' + item['tot_select']
            if step > 2:
                content += '\nWHERE clause: ' + item['tot_where']
            if step > 3:
                content += '\nGROUP BY clause: ' + item['tot_group_by']
            if step > 4:
                content += '\nORDER BY clause: ' + item['tot_order_by']
            if args.gpt in GPT_CHAT_MODELS:
                prompt.append({'role': 'user', 'content': content})
                if TOT_CLAUSES[step] in item:
                    prompt.append({'role': 'assistant', 'content': item[TOT_CLAUSES[step]]})
            elif args.gpt in GPT_COMPLETION_MODELS:
                pass

        if args.gpt in GPT_CHAT_MODELS:
            prompt = [{'role': 'system', 'content': instructions[step]}]
        elif args.gpt in GPT_COMPLETION_MODELS:
            prompt = ''
        else:
            raise ValueError(f'unknown GPT model {args.gpt}')
        for shot in shots:
            add_item(shot)
        add_item(prev_result)
        return prompt

    def get_prompt_tot_evaluate(self, args, cur_results, beam_size):
        db_id, question, sqls = cur_results[0]['db_id'], cur_results[0]['question'], []
        for i, cur_result in enumerate(cur_results):
            sql = cur_result['tot_select']
            if 'tot_from' in cur_result:
                sql += ' ' + cur_result['tot_from']
            if 'tot_where' in cur_result and cur_result['tot_where'].startswith('WHERE '):
                sql += ' ' + cur_result['tot_where']
            if 'tot_group_by' in cur_result and cur_result['tot_group_by'].startswith('GROUP BY '):
                sql += ' ' + cur_result['tot_group_by']
            if 'tot_order_by' in cur_result and cur_result['tot_order_by'].startswith('ORDER BY '):
                sql += ' ' + cur_result['tot_order_by']
            sqls.append(str(i + 1) + '. ' + sql)
        if args.gpt in GPT_CHAT_MODELS:
            prompt = [
                {'role': 'system', 'content': f'Given the database schema and the question, you need to determine the top-{beam_size} among {len(cur_results)} unfinished SQL queries to solve the question. Please print SQL IDs in the last line.'},
                {'role': 'user', 'content': f'Database schema:\n{self.db_prompts[db_id][args.content]}\nQuestion: {question}\nUnfinished SQL queries:\n' + '\n'.join(sqls)}
            ]
        elif args.gpt in GPT_COMPLETION_MODELS:
            prompt = ''
            pass
        else:
            raise ValueError(f'unknown GPT model {args.gpt}')
        return prompt

    def get_prompt_phase_1(self, args, question=None, shots=[]):
        if args.gpt in GPT_CHAT_MODELS:
            prompt = [{'role': 'system', 'content': 'You need to translate the question into the SQL query.'}]
            for shot in shots:
                prompt.append({'role': 'user', 'content': shot['question']})
                prompt.append({'role': 'assistant', 'content': shot['pseudo_query']})
                prompt.append({'role': 'user', 'content': f"Now given the database schema:\n{self.db_prompts[shot['db_id']][args.content]}\nCorrect your answer."})
                prompt.append({'role': 'assistant', 'content': shot['query']})
            if question:
                prompt.append({'role': 'user', 'content': question})
        elif args.gpt in GPT_COMPLETION_MODELS:
            prompt = ''
            for shot in shots:
                prompt += 'Translate the natural utterance into the SQL query: ' + shot['question'] + '\n'
                prompt += shot['pseudo_query'] + '\n'
                prompt += 'Now given the database schema:\n'
                prompt += self.db_prompts[shot['db_id']][args.content] + '\n'
                prompt += 'Correct your answer.\n'
                prompt += shot['query'] + '\n'
            if question:
                prompt += f'Translate the natural utterance into the SQL query: {question}\nSELECT'
        else:
            raise ValueError(f'unknown GPT model {args.gpt}')
        return prompt

    def get_prompt_phase_2(self, previous_prompt, pseudo_query, db_id):
        prompt = previous_prompt
        if isinstance(prompt, list):
            prompt.append({'role': 'assistant', 'content': pseudo_query})
            prompt.append({'role': 'user', 'content': f'Now given the database schema:\n{self.db_prompts[db_id][args.content]}\nCorrect your answer.'})
        else:
            assert prompt[-6:] == 'SELECT'
            prompt = prompt[:-6]
            assert prompt[-1] == '\n'
            prompt += pseudo_query + '\n'
            prompt += 'Now given the database schema:\n'
            prompt += self.db_prompts[db_id][args.content] + '\n'
            prompt += 'Correct your answer.\n'
            prompt += 'SELECT'
        return prompt

    def is_valid_shots(self, shots, args, request=None):
        if request:
            assert request in ['iue', 'select', 'from', 'where', 'group_by', 'order_by']
            for shot in shots:
                if request == 'iue' and shot['tot_iue'].lower() not in SET_OPS:
                    return False
                if request == 'where' and not shot['tot_where'].startswith('WHERE '):
                    return False
                if request == 'group_by' and not shot['tot_group_by'].startswith('GROUP BY '):
                    return False
                if request == 'order_by' and not shot['tot_order_by'].startswith('ORDER BY '):
                    return False
        prompt = self.get_prompt_phase_1(args, shots=shots) if args.two_phase else self.get_prompt(args, shots=shots)
        prompt_len = len(prompt) if isinstance(prompt, str) else sum([len(message['content']) for message in prompt])
        max_len = 15000 if args.gpt == 'code-davinci-002' else 7500
        return prompt_len < max_len * len(shots) / (args.cluster_num + args.dynamic_num)

    def get_static_shots(self, dataset, args, request=None):
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
            if self.is_valid_shots(shots, args, request):
                return shots

    def get_dynamic_shots(self, encoding, dataset, args):
        if args.zero_shot or args.dynamic_num == 0:
            return []
        scores = util.cos_sim(encoding, [example[args.encoding + '_encoding'] for example in dataset]).squeeze(0).tolist()
        scores = sorted(enumerate(scores), key=lambda x: -x[1])
        shots = []
        for item in scores:
            shots.append(dataset[item[0]])
            if not self.is_valid_shots(shots, args):
                shots.pop()
            elif len(shots) == args.dynamic_num:
                break
        return shots


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


if __name__ == '__main__':
    from util.arg import main_args

    def print_prompt(prompt):
        if isinstance(prompt, str):
            print(prompt)
        else:
            for message in prompt:
                print('role:', message['role'])
                print('content:')
                print(message['content'])
                print()

    args = main_args()
    print('log path:', args.log_path)
    prompt_maker = PromptMaker(args=args)
    db_id = input('db: ')
    question = 'List all items in the table.'
    pseudo_query = 'SELECT * FROM table'
    if args.two_phase:
        prompt_phase_1 = prompt_maker.get_prompt_phase_1(args, question)
        print_prompt(prompt_phase_1)
        prompt_phase_2 = prompt_maker.get_prompt_phase_2(prompt_phase_1, pseudo_query, db_id)
        print_prompt(prompt_phase_2)
    else:
        print_prompt(prompt_maker.get_prompt(args, db_id, question))
