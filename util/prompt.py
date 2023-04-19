import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
import sqlite3
from sentence_transformers import util
from util.constant import GPT_CHAT_MODELS, GPT_COMPLETION_MODELS


class PromptMaker:
    def __init__(self, args):
        with open(os.path.join('data', args.dataset, 'tables.json'), 'r', encoding='utf-8') as file:
            dbs = json.load(file)
        self.db_prompts = {}
        for db in dbs:
            db_id = db['db_id']
            tabs = db['table_names_original']
            cols = db['column_names_original']
            self.db_prompts[db_id] = ''
            for i in range(len(tabs)):
                if args.api_doc:
                    self.db_prompts[db_id] += f"# {tabs[i]}({', '.join(col[1] for col in cols if col[0] == i)})\n"
                else:
                    self.db_prompts[db_id] += f'create table {tabs[i]} (\n'
                    for j in range(len(cols)):
                        if cols[j][0] == i:
                            self.db_prompts[db_id] += f"    {cols[j][1]} {db['column_types'][j]}"
                            if args.pf == 'eoc':
                                if j in db['primary_keys']:
                                    self.db_prompts[db_id] += ' primary key'
                                for fk in db['foreign_keys']:
                                    if fk[0] == j:
                                        self.db_prompts[db_id] += f' references {tabs[cols[fk[1]][0]]}({cols[fk[1]][1]})'
                            self.db_prompts[db_id] += ',\n'
                    if args.pf == 'eot':
                        pks = [cols[pk][1] for pk in db['primary_keys'] if cols[pk][0] == i]
                        if len(pks) > 0:
                            self.db_prompts[db_id] += f"    primary key ({', '.join(pks)}),\n"
                        for fk in db['foreign_keys']:
                            if cols[fk[0]][0] == i:
                                self.db_prompts[db_id] += f'    foreign key ({cols[fk[0]][1]}) references {tabs[cols[fk[1]][0]]}({cols[fk[1]][1]}),\n'
                    self.db_prompts[db_id] = self.db_prompts[db_id][:-2] + '\n)\n'
                if args.content > 0:
                    conn = sqlite3.connect(os.path.join('data', args.dataset, 'database', db_id, db_id + '.sqlite'))
                    conn.row_factory = dict_factory
                    cursor = conn.cursor()
                    db_contents = cursor.execute(f'SELECT * FROM {tabs[i]} LIMIT {args.content}').fetchall()
                    self.db_prompts[db_id] += '/*\n'
                    self.db_prompts[db_id] += f"{len(db_contents)} example row{'s' if len(db_contents) > 1 else ''} from table {tabs[i]}:\n"
                    self.db_prompts[db_id] += '\t'.join([col[1] for col in cols if col[0] == i]) + '\n'
                    for record in db_contents:
                        self.db_prompts[db_id] += '\t'.join([str(record[col[1]]) for col in cols if col[0] == i]) + '\n'
                    self.db_prompts[db_id] += '*/\n'
            self.db_prompts[db_id] = self.db_prompts[db_id][:-1]

    def get_prompt(self, args, db_id=None, question=None, shots=[], subqa=None):
        if args.gpt in GPT_CHAT_MODELS:
            prompt = [{'role': 'system', 'content': 'Given the database schema, you need to translate the question into the SQL query.'}]
            for shot in shots:
                prompt.append({'role': 'user', 'content': f"Database schema:\n{self.db_prompts[shot['db_id']]}\nQuestion: {shot['question']}"})
                prompt.append({'role': 'assistant', 'content': shot['query']})
            if subqa:
                prompt.append({'role': 'user', 'content': f"Database schema:\n{self.db_prompts[db_id]}\nQuestion: {subqa['q'][0]}"})
                prompt.append({'role': 'assistant', 'content': subqa['a'][0]})
                for i in range(1, args.subproblem):
                    prompt.append({'role': 'user', 'content': 'Question: ' + subqa['q'][i]})
                    prompt.append({'role': 'assistant', 'content': subqa['a'][i]})
                prompt.append({'role': 'user', 'content': 'Now combine the above results and solve the following question: ' + question})
            elif db_id and question:
                prompt.append({'role': 'user', 'content': f'Database schema:\n{self.db_prompts[db_id]}\nQuestion: {question}'})
        elif args.gpt in GPT_COMPLETION_MODELS:
            prompt = ''
            for shot in shots:
                prompt += 'Given the database schema:\n'
                prompt += self.db_prompts[shot['db_id']] + '\n'
                prompt += 'Translate the natural utterance into the SQL query: ' + shot['question'] + '\n'
                prompt += shot['query'] + '\n'
            if subqa:
                prompt = 'Given the database schema:\n' + self.db_prompts[db_id] + '\n'
                for i in range(args.subproblem):
                    prompt += 'Translate the natural utterance into the SQL query: ' + subqa['q'][i] + '\n'
                    prompt += subqa['a'][i] + '\n'
                prompt += 'Now combine the above results and translate the following natural utterance into the SQL query: ' + question + '\n'
                prompt += 'SELECT'
            elif db_id and question:
                prompt += 'Given the database schema:\n'
                prompt += self.db_prompts[db_id] + '\n'
                prompt += 'Translate the natural utterance into the SQL query: ' + question + '\n'
                prompt += 'SELECT'
        else:
            raise ValueError(f'unknown GPT model {args.gpt}')
        return prompt

    def get_prompt_split_problem(self, args, question):
        if args.gpt in GPT_CHAT_MODELS:
            return [
                {'role': 'system', 'content': f'You need to split the problem into exactly {args.subproblem} subproblems.'},
                {'role': 'user', 'content': 'Problem: ' + question}
            ]
        if args.gpt in GPT_COMPLETION_MODELS:
            return f'Split the problem into exactly {args.subproblem} subproblems: {question}\n1.'
        raise ValueError(f'unknown GPT model {args.gpt}')

    def get_prompt_remove_context_dependency(self, args, subproblems):
        content = ''
        for i, subproblem in enumerate(subproblems):
            content += str(i + 1) + '. ' + subproblem + '\n'
        if args.gpt in GPT_CHAT_MODELS:
            return [
                {'role': 'system', 'content': 'You need to rewrite the second sentence to remove the context dependency between 2 sentences.'},
                {'role': 'user', 'content': content.strip()}
            ]
        if args.gpt in GPT_COMPLETION_MODELS:
            return 'Rewrite the second sentence to remove the context dependency between 2 sentences:\n' + content + 'Result:'
        raise ValueError(f'unknown GPT model {args.gpt}')

    def get_prompt_phase_1(self, args, question=None, shots=[]):
        if args.gpt in GPT_CHAT_MODELS:
            prompt = [{'role': 'system', 'content': 'You need to translate the question into the SQL query.'}]
            for shot in shots:
                prompt.append({'role': 'user', 'content': shot['question']})
                prompt.append({'role': 'assistant', 'content': shot['pseudo_query']})
                prompt.append({'role': 'user', 'content': f"Now given the database schema:\n{self.db_prompts[shot['db_id']]}\nCorrect your answer."})
                prompt.append({'role': 'assistant', 'content': shot['query']})
            if question:
                prompt.append({'role': 'user', 'content': question})
        elif args.gpt in GPT_COMPLETION_MODELS:
            prompt = ''
            for shot in shots:
                prompt += 'Translate the natural utterance into the SQL query: ' + shot['question'] + '\n'
                prompt += shot['pseudo_query'] + '\n'
                prompt += 'Now given the database schema:\n'
                prompt += self.db_prompts[shot['db_id']] + '\n'
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
            prompt.append({'role': 'user', 'content': f'Now given the database schema:\n{self.db_prompts[db_id]}\nCorrect your answer.'})
        else:
            assert prompt[-6:] == 'SELECT'
            prompt = prompt[:-6]
            assert prompt[-1] == '\n'
            prompt += pseudo_query + '\n'
            prompt += 'Now given the database schema:\n'
            prompt += self.db_prompts[db_id] + '\n'
            prompt += 'Correct your answer.\n'
            prompt += 'SELECT'
        return prompt

    def is_valid_shots(self, shots, args):
        prompt = self.get_prompt_phase_1(args, shots=shots) if args.two_phase else self.get_prompt(args, shots=shots)
        prompt_len = len(prompt) if isinstance(prompt, str) else sum([len(message['content']) for message in prompt])
        max_len = 15000 if args.gpt == 'code-davinci-002' else 7500
        return prompt_len < max_len * len(shots) / (args.cluster_num + args.dynamic_num)

    def get_static_shots(self, dataset, args):
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
            if self.is_valid_shots(shots, args):
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
    if args.subproblem > 1:
        subqa = {'q': [], 'a': []}
        for i in range(args.subproblem):
            subqa['q'].append('This is the subproblem ' + str(i + 1) + '.')
            subqa['a'].append('SELECT * FROM table' + str(i + 1))
        print_prompt(prompt_maker.get_prompt_split_problem(args, question))
        print_prompt(prompt_maker.get_prompt_remove_context_dependency(args, subqa['q']))
        print_prompt(prompt_maker.get_prompt(args, db_id, question, subqa=subqa))
    elif args.two_phase:
        prompt_phase_1 = prompt_maker.get_prompt_phase_1(args, question)
        print_prompt(prompt_phase_1)
        prompt_phase_2 = prompt_maker.get_prompt_phase_2(prompt_phase_1, pseudo_query, db_id)
        print_prompt(prompt_phase_2)
    else:
        print_prompt(prompt_maker.get_prompt(args, db_id, question))
