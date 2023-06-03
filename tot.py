import json
import os
from util.arg import tot_args
from util.constant import AGGS, CONDS, OPS, SET_OPS


def parse_sql(sql, db):
    result = parse_sql_unit(sql, db)
    for set_op in SET_OPS:
        if sql[set_op]:
            result += ' ' + set_op.upper() + ' ' + parse_sql(sql[set_op], db)
    return result


def parse_sql_unit(sql_unit, db):
    result = parse_select(sql_unit['select'], db) + ' ' + parse_from(sql_unit['from'], db)
    if sql_unit['where']:
        result += ' ' + parse_where(sql_unit['where'], db)
    if sql_unit['groupBy']:
        result += ' ' + parse_group_by(sql_unit['groupBy'], sql_unit['having'], db)
    if sql_unit['orderBy']:
        result += ' ' + parse_order_by(sql_unit['orderBy'], sql_unit['limit'], db)
    return result


def parse_select(select, db):
    val_units = []
    for item in select[1]:
        val_unit = parse_val_unit(item[1], db)
        if select[0]:
            val_unit = 'DISTINCT ' + val_unit
        if item[0] > 0:
            val_unit = AGGS[item[0]] + '(' + val_unit + ')'
        val_units.append(val_unit)
    return 'SELECT ' + ', '.join(val_units)


def parse_from(from_clause, db):
    table_units = []
    for item in from_clause['table_units']:
        if item[0] == 'table_unit':
            table_units.append(db['table_names_original'][item[1]])
        else:
            table_units.append('(' + parse_sql(item[1], db) + ')')
    result = 'FROM ' + ' JOIN '.join(table_units)
    if from_clause['conds']:
        result += ' ON ' + parse_conds(from_clause['conds'], db)
    return result


def parse_where(where, db):
    return 'WHERE ' + parse_conds(where, db)


def parse_group_by(group_by, having, db):
    col_units = []
    for item in group_by:
        col_units.append(parse_col_unit(item, db))
    result = 'GROUP BY ' + ', '.join(col_units)
    if having:
        result += ' HAVING ' + parse_conds(having, db)
    return result


def parse_order_by(order_by, limit, db):
    val_units = []
    for item in order_by[1]:
        val_units.append(parse_val_unit(item, db))
    result = 'ORDER BY ' + ', '.join(val_units) + ' ' + order_by[0].upper()
    if isinstance(limit, int):
        result += ' LIMIT ' + str(limit)
    return result


def parse_conds(conds, db):
    result = ''
    for cond in conds:
        if isinstance(cond, str):
            result += ' ' + cond.upper() + ' '
            continue
        result += parse_val_unit(cond[2], db)
        if cond[0]:
            assert CONDS[cond[1]] in ['IN', 'LIKE']
            result += ' NOT'
        result += ' ' + CONDS[cond[1]] + ' '
        if isinstance(cond[3], dict):
            result += '(' + parse_sql(cond[3], db) + ')'
        elif isinstance(cond[3], list):
            result += parse_col_unit(cond[3], db)
        else:
            result += parse_value(cond[3])
        if CONDS[cond[1]] == 'BETWEEN':
            result += ' AND ' + parse_value(cond[4])
    return result


def parse_val_unit(val_unit, db):
    result = parse_col_unit(val_unit[1], db)
    if val_unit[0] > 0:
        result += ' ' + OPS[val_unit[0]] + ' ' + parse_col_unit(val_unit[2], db)
    return result


def parse_col_unit(col_unit, db):
    col_id = col_unit[1]
    if col_id > 0:
        table_name = db['table_names_original'][db['column_names_original'][col_id][0]]
        column_name = db['column_names_original'][col_id][1]
        result = table_name + '.' + column_name
    else:
        result = '*'
    if col_unit[2]:
        result = 'DISTINCT ' + result
    if col_unit[0] > 0:
        result = AGGS[col_unit[0]] + '(' + result + ')'
    return result


def parse_value(value):
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    assert isinstance(value, float)
    return str(int(value)) if value == int(value) else str(value)


args = tot_args()
with open(os.path.join('data', args.dataset, 'tables.json'), 'r', encoding='utf-8') as file:
    dbs = {db['db_id']: db for db in json.load(file)}
with open(os.path.join('data', args.dataset, 'train.json'), 'r', encoding='utf-8') as file:
    dataset = json.load(file)
for example in dataset:
    db = dbs[example['db_id']]
    sql = example['sql']
    for set_op in SET_OPS:
        if sql[set_op]:
            example['iue'] = set_op.upper()
            break
    else:
        example['iue'] = 'No set operator is needed.'
    example['tot_select'] = parse_select(sql['select'], db)
    example['tot_from'] = parse_from(sql['from'], db)
    example['tot_where'] = parse_where(sql['where'], db) if sql['where'] else 'The WHERE clause is not needed.'
    example['tot_group_by'] = parse_group_by(sql['groupBy'], sql['having'], db) if sql['groupBy'] else 'The GROUP BY clause is not needed.'
    example['tot_order_by'] = parse_order_by(sql['orderBy'], sql['limit'], db) if sql['orderBy'] else 'The ORDER BY clause is not needed.'
    if example['iue'].lower() in SET_OPS:
        set_op = example['iue'].lower()
        example['tot_iue_select'] = parse_select(sql[set_op]['select'], db)
        example['tot_iue_from'] = parse_from(sql[set_op]['from'], db)
        example['tot_iue_where'] = parse_where(sql[set_op]['where'], db) if sql[set_op]['where'] else 'The WHERE clause is not needed.'
        example['tot_iue_group_by'] = parse_group_by(sql[set_op]['groupBy'], sql[set_op]['having'], db) if sql[set_op]['groupBy'] else 'The GROUP BY clause is not needed.'
        example['tot_iue_order_by'] = parse_order_by(sql[set_op]['orderBy'], sql[set_op]['limit'], db) if sql[set_op]['orderBy'] else 'The ORDER BY clause is not needed.'
with open(os.path.join('data', args.dataset, 'train.json'), 'w', encoding='utf-8') as file:
    json.dump(dataset, file, ensure_ascii=False, indent=4)
