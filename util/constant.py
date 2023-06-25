GPT_CHAT_MODELS = ['gpt-3.5-turbo', 'gpt-4']

GPT_COMPLETION_MODELS = ['code-davinci-002', 'text-davinci-003']

MAX_LENS = {
    'gpt-3.5-turbo': 7500,
    'gpt-4': 15000,
    'code-davinci-002': 15000,
    'text-davinci-003': 7500
}

SPEECH_API_TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJoY3o5OSIsImlhdCI6MTY3OTY0Nzc0NywiZXhwIjoxNjgyMjM5NzQ3LCJuYW1lX2NuIjoiXHU1ZjIwXHU2NjU3XHU3ZmMwIiwidXNlcm5hbWUiOiJoY3o5OSIsIm9yZyI6InNqdHUifQ.AYi6YCKqgRoSUbDFaDWH25RwVri79BlFiaPsDDvLXNs'

AGGS = [None, 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']

CONDS = [None, 'BETWEEN', '=', '>', '<', '>=', '<=', '!=', 'IN', 'LIKE']

OPS = [None, '-', '+', '*', '/']

SET_OPS = ['intersect', 'union', 'except']

TOT_CLAUSES = [
    'tot_iue',
    'tot_select',
    'tot_where',
    'tot_group_by',
    'tot_order_by',
    'tot_from'
]

TOT_INSTRUCTIONS = [
    'Given the database schema, you need to translate the question into the SQL query. In this step, you need to determine the set operator (INTERSECT / UNION / EXCEPT) in the SQL query.',
    'Given the database schema, you need to translate the question into the SQL query. In this step, you need to determine the SELECT clause.',
    'Given the database schema, you need to translate the question into the SQL query. In previous steps, some clauses have been determined. In this step, you need to determine the WHERE clause.',
    'Given the database schema, you need to translate the question into the SQL query. In previous steps, some clauses have been determined. In this step, you need to determine the GROUP BY clause (including the HAVING clause if needed).',
    'Given the database schema, you need to translate the question into the SQL query. In previous steps, some clauses have been determined. In this step, you need to determine the ORDER BY clause (including the LIMIT clause if needed).',
    'Given the database schema, you need to translate the question into the SQL query. In previous steps, some clauses have been determined. In this step, you need to determine the FROM clause (including the ON clause if needed).'
]

TOT_STOPS = [
    None,
    ['FROM', 'WHERE', 'GROUP BY', 'ORDER BY'],
    ['SELECT', 'FROM', 'GROUP BY', 'ORDER BY'],
    ['SELECT', 'FROM', 'WHERE', 'ORDER BY'],
    ['SELECT', 'FROM', 'WHERE', 'GROUP BY'],
    ['SELECT', 'WHERE', 'GROUP BY', 'ORDER BY']
]
