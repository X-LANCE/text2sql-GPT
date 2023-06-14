import json
import os
from util.arg import multiturn_args
from util.gpt import get_response
from util.prompt import PromptMaker


def generate_single_turn_dataset(args, choice):
    with open(os.path.join('data', args.dataset, choice + '_multiturn.json'), 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    with open(os.path.join('data', args.dataset, 'shot.json'), 'r', encoding='utf-8') as file:
        shots = json.load(file)
    output_path = os.path.join('data', args.dataset, choice + '.json')
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as file:
            result = json.load(file)
    else:
        result = []
    for i, example in enumerate(dataset):
        print(f'Generating example {i} ...')
        if len(result) > 0 and i <= result[-1]['e_id']:
            continue
        for shot in shots:
            if i == shot['e_id'] and choice == 'train':
                questions = shot['q']
                break
        else:
            questions_multiturn = [turn['utterance'] for turn in example['interaction']]
            if questions_multiturn[-1].lower().startswith('yes please'):
                questions_multiturn.pop()
            if questions_multiturn[-1].lower() == 'yes':
                questions_multiturn.pop()
            i_len = len(questions_multiturn)
            if i_len > 1 or (args.dataset == 'cosql' and '|' in questions_multiturn[0]):
                prompt = PromptMaker.get_prompt_remove_dependency(args.gpt, questions_multiturn, shots)
                temperature = 0
                while 1:
                    try:
                        questions = get_response(prompt, args, max_tokens=750, temperature=temperature).strip().split('\n')
                        if i_len == 1 and not questions[0].startswith('1. '):
                            questions[0] = '1. ' + questions[0]
                        if questions_multiturn[0].lower().startswith('yes please') and len(questions) < i_len:
                            questions.insert(0, '1. Yes Please.')
                        if example['interaction'][-1]['utterance'] == '* I have left the chat *' and len(questions) < i_len:
                            questions.append(str(i_len) + '. * I have left the chat *')
                        elif example['interaction'][-1]['utterance'] == 'Yes' and len(questions) < len(example['interaction']):
                            questions.append(str(len(example['interaction'])) + '. Yes.')
                        for j in range(len(questions)):
                            prefix = str(j + 1) + '. '
                            assert questions[j].startswith(prefix)
                            questions[j] = questions[j][len(prefix):]
                        break
                    except:
                        temperature += 0.1
            else:
                questions = [turn['utterance'] for turn in example['interaction']]
        for j, turn in enumerate(example['interaction']):
            result.append({
                'e_id': i,
                'db_id': example['database_id'],
                'query': turn['query'],
                'question': questions[j],
                'sql': turn['sql']
            })
        if (i + 1) % 10 == 0 or i == len(dataset) - 1:
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(result, file, ensure_ascii=False, indent=4)


args = multiturn_args()
generate_single_turn_dataset(args, 'train')
generate_single_turn_dataset(args, 'dev')
