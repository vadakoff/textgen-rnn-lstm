import re


def read_tokens(path: str):
    with open(path, 'r', encoding='utf-8') as file:
        data = file.read()
        return re.split(r'\s+', data)
