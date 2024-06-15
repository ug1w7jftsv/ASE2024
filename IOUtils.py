import json
import os

from IR.DataSets.LinkSet import LinkSet


def read_json_file(file_path):
    with open(file_path, encoding='utf8') as json_file:
        return json.load(json_file)


def save_as_text_file(save_dir, file_name, file_lines: list[str], file_prefix=""):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    _file_name = file_prefix + file_name
    _file_path = os.path.join(save_dir, _file_name)
    with open(_file_path, 'w', encoding='utf8') as _file:
        _file.write("\r\n".join(file_lines))


def save_as_json_file(save_dir, file_name, obj, file_prefix=""):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    _file_name = file_prefix + file_name
    _file_path = os.path.join(save_dir, _file_name)
    with open(_file_path, 'w', encoding='utf8') as _file:
        json.dump(obj, _file, indent=2, ensure_ascii=False)


def read_text_line(path, encode='utf8') -> list:
    text_lines = []
    with open(path, encoding=encode) as text:
        for line_num, line_text in enumerate(text):
            text_lines.append(line_text.strip())
    return text_lines


def read_stopwords(path) -> set:
    return set(read_text_line(path))


def read_all_stopwords(dir_path) -> set:
    stopwords = set()
    for stopwords_file in os.listdir(dir_path):
        stopwords_path = os.path.join(dir_path, stopwords_file)
        if os.path.isfile(stopwords_path):
            stopwords = stopwords | read_stopwords(stopwords_path)
    return stopwords


def read_json_link_set(issue_path, commit_path, link_path) -> LinkSet:
    issues = read_json_file(issue_path)
    commits = read_json_file(commit_path)
    links = [tuple(link) for link in read_json_file(link_path)]
    link_set = LinkSet(issues, commits, links)
    return link_set
