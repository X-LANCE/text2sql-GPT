import json
import os
import sqlite3


class Prompt:
    def __init__(self, args, table_path='data/tables.json', db_dir='data/database'):
        with open(table_path, 'r', encoding='utf-8') as file:
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
                    conn = sqlite3.connect(os.path.join(db_dir, db_id, db_id + '.sqlite'))
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

    def get_prompt(self, db_id=None, question=None, shots=[]):
        prompt = ''
        for shot in shots:
            prompt += 'Given the database schema:\n'
            prompt += self.db_prompts[shot['db_id']] + '\n'
            prompt += 'Translate the natural utterance into the SQL query: ' + shot['question'] + '\n'
            prompt += shot['query'] + '\n'
        if db_id and question:
            prompt += 'Given the database schema:\n'
            prompt += self.db_prompts[db_id] + '\n'
            prompt += 'Translate the natural utterance into the SQL query: ' + question + '\n'
            prompt += 'SELECT'
        return prompt


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from util.arg import main_args
    args = main_args()
    print('log path:', args.log_path)
    prompt = Prompt(args=args)
    db_id = input('db: ')
    question = input('question: ')
    print(prompt.get_prompt(db_id, question))
