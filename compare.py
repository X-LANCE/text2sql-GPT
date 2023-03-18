import json


def get_ex_failed(log_path):
    ex_failed = set()
    with open(f'log/{log_path}/dev.txt', 'r', encoding='utf-8') as file:
        while 1:
            content = file.readline().strip()
            if content not in ['EX fail', 'EM fail']:
                break
            id = int(file.readline().strip()[4:])
            if content == 'EX fail':
                ex_failed.add(id)
            for _ in range(3):
                file.readline()
    return ex_failed


log_path1 = input('log path 1: ')
log_path2 = input('log path 2: ')
ex_failed1 = get_ex_failed(log_path1)
ex_failed2 = get_ex_failed(log_path2)
with open('data/dev.json', 'r', encoding='utf-8') as file:
    dataset_size = len(json.load(file))
eval_matrix = {
    'TT': 0,
    'TF': 0,
    'FT': 0,
    'FF': 0
}
for i in range(dataset_size):
    eval1 = 'F' if i in ex_failed1 else 'T'
    eval2 = 'F' if i in ex_failed2 else 'T'
    eval = eval1 + eval2
    eval_matrix[eval] += 1
for eval in eval_matrix:
    eval_matrix[eval] = str(round(eval_matrix[eval] / dataset_size * 100, 2)) + '%'
print(''.rjust(4) + '2 T'.rjust(8) + '2 F'.rjust(8))
print('1 T'.ljust(4), end='')
print(eval_matrix['TT'].rjust(8), end='')
print(eval_matrix['TF'].rjust(8))
print('1 F'.ljust(4), end='')
print(eval_matrix['FT'].rjust(8), end='')
print(eval_matrix['FF'].rjust(8))
