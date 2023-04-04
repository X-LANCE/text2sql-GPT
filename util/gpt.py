import json
import openai
import os
import requests
import time
import ujson
from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
from util.constant import SPEECH_API_TOKEN


class OpenAIModel(Enum):
    ADA_001 = 'text-ada-001'
    BABBAGE_001 = 'text-babbage-001'
    DAVINCI_003 = 'text-davinci-003'
    EMBEDDING = 'text-embedding-ada-002'
    FREE_CHATGPT_MODEL = 'text-davinci-002-render-sha'
    PAID_CHATGPT_MODEL = 'text-davinci-002-render-paid'
    TURBO = 'gpt-3.5-turbo'


class ChatRole(Enum):
    ROBOT = 'assistant'
    USER = 'user'
    SYSTEM = 'system'


class CompletionParams(BaseModel):
    model: OpenAIModel = Field(default=OpenAIModel.BABBAGE_001)
    user: str = Field(default=None)
    max_tokens: int = Field(default=100)
    temperature: float = Field(default=0.0)
    presence_penalty: float = Field(default=0.0)
    frequency_penalty: float = Field(default=0.0)
    n: Optional[int] = Field(default=1, const=True)
    stream: Optional[bool] = Field(default=False)
    top_p: int = Field(default=1, const=True)
    logit_bias: Optional[Dict[str, float]] = Field(default=None)
    stop: Optional[Union[str, List[str]]] = Field(default=None)


class TextCompletionParams(CompletionParams):
    prompt: str = Field(default='This is a test!')
    suffix: Optional[str] = Field(default=None)
    logprobs: Optional[int] = Field(default=None, le=5)
    echo: Optional[bool] = Field(default=False, const=True)
    best_of: Optional[int] = Field(default=1, const=True)


class ChatMessage(BaseModel):
    content: str = Field(default='')
    role: ChatRole = Field(default=ChatRole.USER)


class ChatCompletionParams(CompletionParams):
    messages: List[ChatMessage] = Field()

    class Config:
        json_loads = ujson.loads
        json_dumps = ujson.dumps
        schema_extra = {
            'example': {
                'model': OpenAIModel.TURBO,
                'user': 'test-user',
                'messages': [{'role': 'user', 'content': '你好，今天天气不错！'}],
                'max_tokens': 1000,
                'temperature': 0.0,
                'presence_penalty': 0.0,
                'frequency_penalty': 0.0
            }
        }


def get_response(prompt, args):
    if args.speech_api:
        post_data = {
            'model': args.gpt,
            'max_tokens': 150,
            'temperature': 0,
            'top_p': 1,
            'frequency_penalty': 0,
            'presence_penalty': 0
        }
        if isinstance(prompt, str):
            url = 'http://54.193.55.85:10030/v1/completions?use_cache=false'
            post_data['prompt'] = prompt
            post_data['stop'] = [';', '\n\n', 'Given', 'Translate']
        else:
            url = 'http://54.193.55.85:10030/v1/completions/chat?use_cache=false'
            post_data['messages'] = prompt
            post_data['stop'] = [';']
        while 1:
            try:
                response = requests.post(url, json=post_data, headers={'llm-token': SPEECH_API_TOKEN})
                result = json.loads(response.text)['result']
                return result['text'] if isinstance(prompt, str) else result['message']['content']
            except:
                print('Retrying ...')
                time.sleep(10)
    openai.api_key = os.getenv('OPENAI_API_KEY')
    while 1:
        if isinstance(prompt, str):
            try:
                response = openai.Completion.create(
                    model=args.gpt,
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=[';', '\n\n', 'Given', 'Translate']
                )
                return response['choices'][0]['text']
            except:
                print('Retrying ...')
                time.sleep(10)
        else:
            try:
                response = openai.ChatCompletion.create(
                    model=args.gpt,
                    messages=prompt,
                    max_tokens=150,
                    temperature=0,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=[';']
                )
                return response['choices'][0]['message']['content']
            except:
                print('Retrying ...')
                time.sleep(10)
