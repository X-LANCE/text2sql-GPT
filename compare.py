import json


def get_ex_failed(log_path):
    ex_failed = {}
    with open(f'log/{log_path}/dev.txt', 'r', encoding='utf-8') as file:
        while 1:
            content = file.readline().strip()
            if content not in ['EX fail', 'EM fail']:
                break
            if content == 'EM fail':
                for _ in range(4):
                    file.readline()
                continue
            id = int(file.readline().strip()[4:])
            ex_failed[id] = file.readline() + file.readline()
            file.readline()
    return ex_failed


def compare(dataset, log_path1, log_path2, ex_failed1, ex_failed2, of):
    for id in ex_failed1:
        if id not in ex_failed2:
            of.write(f'{log_path1} wrong, {log_path2} correct\n')
            of.write(f"db: {dataset[id]['db_id']}\n")
            of.write(f"question: {dataset[id]['question']}\n")
            of.write(ex_failed1[id] + '\n')


log_path1 = input('log path 1: ')
log_path2 = input('log path 2: ')
ex_failed1 = get_ex_failed(log_path1)
ex_failed2 = get_ex_failed(log_path2)
with open('data/dev.json', 'r', encoding='utf-8') as file:
    dataset = json.load(file)
of = open(f'log/{log_path1}-{log_path2}.txt', 'w', encoding='utf-8')
compare(dataset, log_path1, log_path2, ex_failed1, ex_failed2, of)
compare(dataset, log_path2, log_path1, ex_failed2, ex_failed1, of)
of.close()
