import json
import os
from itertools import combinations
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer, util
from util.arg import cot_args


def get_tables_in_sql(sql, db):
    tables = get_tables_in_sql_unit(sql, db)
    for set_op in ['intersect', 'union', 'except']:
        if sql[set_op]:
            tables.update(get_tables_in_sql(sql[set_op], db))
    return tables


def get_tables_in_sql_unit(sql_unit, db):
    tables = set()
    for table_unit in sql_unit['from']['table_units']:
        if table_unit[0] == 'table_unit':
            tables.add(db['table_names_original'][table_unit[1]])
        else:
            tables.update(get_tables_in_sql(table_unit[1], db))
    tables.update(get_tables_in_conds(sql_unit['where'][::2], db))
    tables.update(get_tables_in_conds(sql_unit['having'][::2], db))
    return tables


def get_tables_in_conds(conds, db):
    tables = set()
    for cond in conds:
        if isinstance(cond[3], dict):
            tables.update(get_tables_in_sql(cond[3], db))
    return tables


def get_columns_in_sql(sql, db):
    columns = get_columns_in_sql_unit(sql, db)
    for set_op in ['intersect', 'union', 'except']:
        if sql[set_op]:
            columns.update(get_columns_in_sql(sql[set_op], db))
    return columns


def get_columns_in_sql_unit(sql_unit, db):
    columns = set()
    for val_unit in sql_unit['select'][1]:
        columns.update(get_columns_in_val_unit(val_unit[1], db))
    for table_unit in sql_unit['from']['table_units']:
        if table_unit[0] == 'sql':
            columns.update(get_columns_in_sql(table_unit[1], db))
    columns.update(get_columns_in_conds(sql_unit['where'][::2], db))
    columns.update(get_columns_in_conds(sql_unit['having'][::2], db))
    if sql_unit['orderBy']:
        for val_unit in sql_unit['orderBy'][1]:
            columns.update(get_columns_in_val_unit(val_unit, db))
    return columns


def get_columns_in_conds(conds, db):
    columns = set()
    for cond in conds:
        columns.update(get_columns_in_val_unit(cond[2], db))
        if isinstance(cond[3], dict):
            columns.update(get_columns_in_sql(cond[3], db))
        elif isinstance(cond[3], list):
            columns.update(get_columns_in_col_unit(cond[3], db))
    return columns


def get_columns_in_val_unit(val_unit, db):
    if isinstance(val_unit, dict):
        return get_columns_in_sql(val_unit, db)
    elif isinstance(val_unit, list):
        columns = get_columns_in_col_unit(val_unit[1], db)
        if val_unit[2]:
            columns.update(get_columns_in_col_unit(val_unit[2], db))
        return columns
    return set()


def get_columns_in_col_unit(col_unit, db):
    col_id = col_unit[1]
    if col_id == 0:
        return set()
    table_name = db['table_names_original'][db['column_names_original'][col_id][0]]
    column_name = db['column_names_original'][col_id][1]
    return {table_name + '.' + column_name}


def get_values_in_sql(sql):
    values = get_values_in_sql_unit(sql)
    for set_op in ['intersect', 'union', 'except']:
        if sql[set_op]:
            values.update(get_values_in_sql(sql[set_op]))
    return values


def get_values_in_sql_unit(sql_unit):
    values = set()
    for table_unit in sql_unit['from']['table_units']:
        if table_unit[0] == 'sql':
            values.update(get_values_in_sql(table_unit[1]))
    values.update(get_values_in_conds(sql_unit['where'][::2]))
    values.update(get_values_in_conds(sql_unit['having'][::2]))
    if isinstance(sql_unit['limit'], int):
        values.add(str(sql_unit['limit']))
    return values


def get_values_in_conds(conds):
    values = set()
    for cond in conds:
        for value in cond[3:]:
            if isinstance(value, dict):
                values.update(get_values_in_sql(value))
            elif isinstance(value, str):
                values.add(value.strip('"').strip("'"))
            elif isinstance(value, int):
                values.add(str(value))
            elif isinstance(value, float):
                if value == int(value):
                    values.add(str(int(value)))
                else:
                    values.add(str(value))
    return values


args = cot_args()
sentence_encoder = SentenceTransformer(os.path.join('plm', args.plm))
with open(os.path.join('data', args.dataset, 'tables.json'), 'r', encoding='utf-8') as file:
    dbs = {db['db_id']: db for db in json.load(file)}
with open(os.path.join('data', args.dataset, 'train.json'), 'r', encoding='utf-8') as file:
    dataset = json.load(file)
for i, example in enumerate(dataset):
    if 'cot' in example:
        continue
    print(f'Generating CoT for example {i} ...')
    words = word_tokenize(example['question'])
    phrases = [' '.join(words[item[0]:item[1]]) for item in combinations(range(len(words) + 1), 2)]
    phrase_encodings = sentence_encoder.encode(
        phrases,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_tensor=True,
        device=args.device
    ).cpu().tolist()
    tables = get_tables_in_sql(example['sql'], dbs[example['db_id']])
    columns = get_columns_in_sql(example['sql'], dbs[example['db_id']])
    values = get_values_in_sql(example['sql'])
    schema_items = []
    for table in tables:
        for column in columns:
            if table in column:
                break
        else:
            schema_items.append((table, 'table'))
    for column in columns:
        schema_items.append((column, 'column'))
    schema_linkings = {}
    for schema_item in schema_items:
        encoding = sentence_encoder.encode(
            schema_item[0],
            batch_size=1,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_tensor=True,
            device=args.device
        ).cpu().tolist()
        scores = util.cos_sim(encoding, phrase_encodings).squeeze(0).tolist()
        phrase = phrases[max(enumerate(scores), key=lambda x: x[1])[0]]
        if phrase not in schema_linkings:
            schema_linkings[phrase] = {'table': [], 'column': []}
        schema_linkings[phrase][schema_item[1]].append(schema_item[0])
    example['cot'] = "Let's think step by step.\n"
    for phrase in schema_linkings:
        example['cot'] += f'According to "{phrase}",'
        if schema_linkings[phrase]['table']:
            example['cot'] += f' tables [{", ".join(schema_linkings[phrase]["table"])}]'
        if schema_linkings[phrase]['column']:
            if example['cot'].endswith(']'):
                example['cot'] += ' and'
            example['cot'] += f' columns [{", ".join(schema_linkings[phrase]["column"])}]'
        example['cot'] += ' may be used.\n'
    if values:
        example['cot'] += f'Values [{", ".join(values)}] may be used.\n'
    example['cot'] += 'So the final answer is:\n'
    example['cot'] += ' '.join(example['query'].strip('\t ;').split())
    if (i + 1) % 100 == 0 or i == len(dataset) - 1:
        with open(os.path.join('data', args.dataset, 'train.json'), 'w', encoding='utf-8') as file:
            json.dump(dataset, file, ensure_ascii=False, indent=4)
