import abc
import configparser
import os
import re
import sys
import time

from tqdm import tqdm

import IOUtils
from LangUtils import has_lang

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config.ini"), encoding="UTF-8")

dataset_base_dir = config.get("path", "dataset_path")
save_base_dir = dataset_base_dir


class BaseTranslator(metaclass=abc.ABCMeta):
    def __init__(self, lang="zh"):
        self.bilingual_table = {
            "zh": "zho_Hans",
            "jp": "jpn_Japn",
            "po": "por_Latn",
            "ko": "kor_Hang"
        }
        self.lang = lang
        self.has_foreign = has_lang[lang]

    @abc.abstractmethod
    def translate_sentences(self, sentences: list[str], indexes: list[int]):
        pass

    def get_bilingual_indexes(self, sentences: list[str]):
        bilingual_indexes = []
        for i in range(len(sentences)):
            sentence = sentences[i]
            if self.has_foreign(sentence):
                bilingual_indexes.append(i)
        return bilingual_indexes

    def translate_issues(self, issues: dict):
        for issue_id in tqdm(issues, desc="Translating issues:", file=sys.stdout, colour="#646cff"):
            summary_sentences = re.split("\n", issues[issue_id]["summary"])
            summary_bilingual_indexes = self.get_bilingual_indexes(summary_sentences)

            description_sentences = re.split("\n", issues[issue_id]["description"])
            description_bilingual_indexes = self.get_bilingual_indexes(description_sentences)
            issues[issue_id] = {
                "summary": "\n".join(self.translate_sentences(summary_sentences, summary_bilingual_indexes)),
                "description": "\n".join(
                    self.translate_sentences(description_sentences, description_bilingual_indexes)),
                "summary_bilingual_indexes": summary_bilingual_indexes,
                "description_bilingual_indexes": description_bilingual_indexes
            }
        return issues

    def translate_commits(self, commits: dict):
        for commit_id in tqdm(commits, desc="Translating commits:", file=sys.stdout, colour="#646cff"):
            message_sentences = re.split("\n", commits[commit_id]["message"])
            message_bilingual_indexes = self.get_bilingual_indexes(message_sentences)

            diff_sentences = re.split("\n", commits[commit_id]["diff"])
            diff_bilingual_indexes = self.get_bilingual_indexes(diff_sentences)
            commits[commit_id] = {
                "message": "\n".join(self.translate_sentences(message_sentences, message_bilingual_indexes)),
                "diff": "\n".join(self.translate_sentences(diff_sentences, diff_bilingual_indexes)),
                "message_bilingual_indexes": message_bilingual_indexes,
                "diff_bilingual_indexes": diff_bilingual_indexes
            }
        return commits

    def translating(self, repo_list: list[str], src_sub_path, tgt_sub_path):
        for repo in repo_list:
            print("\033[31m" + repo + "\033[0m")
            st = time.time()
            source_repo_path = os.path.join(dataset_base_dir, repo, src_sub_path)
            target_repo_path = os.path.join(dataset_base_dir, repo, tgt_sub_path)

            issues = IOUtils.read_json_file(os.path.join(source_repo_path, "issues.json"))
            issues = self.translate_issues(issues)
            IOUtils.save_as_json_file(target_repo_path, "issues.json", issues)

            commits = IOUtils.read_json_file(os.path.join(source_repo_path, "commits.json"))
            commits = self.translate_commits(commits)
            IOUtils.save_as_json_file(target_repo_path, "commits.json", commits)
            print("\033[31mTranslation has been done in {:.1f}s.\033[0m".format(time.time() - st))
