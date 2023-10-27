import json
import openai
import requests
import time


def get_response(prompt, args, max_tokens=150, temperature=0):
    while 1:
        if isinstance(prompt, str):
            try:
                response = openai.Completion.create(
                    model=args.gpt,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=[';', '\n\n', 'Given', 'Question', 'Translate']
                )
                return response['choices'][0]['text']
            except Exception as e:
                if str(e).startswith("This model's maximum context length is"):
                    return None
                print('Retrying ...')
                time.sleep(10)
        else:
            try:
                response = openai.ChatCompletion.create(
                    model=args.gpt,
                    messages=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                return response['choices'][0]['message']['content']
            except Exception as e:
                if str(e).startswith("This model's maximum context length is"):
                    return None
                print('Retrying ...')
                time.sleep(10)
