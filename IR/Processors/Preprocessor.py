import configparser
import os
import re
import sys
import time

import many_stop_words
import nltk
import hanlp
from tqdm import tqdm

import IOUtils
from IR.DataSets.LinkSet import LinkSet

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config.ini"))

nltk.data.path.insert(0, config.get("path", "nltk_resources"))
stopwords_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../stopwords")


class Preprocessor:
    def __init__(self, fo_lang_code=None, use_project_stopwords=False, project_name=""):
        print("Init Preprocessor...")
        start_time = time.time()
        self.fo_lang_code = fo_lang_code
        self.fo_lang_patterns = {
            "zh": "([\u4e00-\u9fff]+)",
            "en": "([a-zA-Z]+)",
            "fr": "([a-zA-ZÀ-ÿ]+)",
            "jp": "([\u0800-\u9fff]+)",
            "ko": "([\uac00-\ud7a3]+)",
            "po": "([a-zA-Z\u0080-\u00ff]+)",
        }
        self.bilingual_lang_patterns = {
            "zh": "[^_a-zA-Z\u4e00-\u9fff]+",
            "en": "[^_a-zA-Z]+",
            "fr": "[^_a-zA-Z\u0080-\u00ff]+",
            "jp": "[^_a-zA-Z\u0800-\u9fff]+",
            "ko": "[^_a-zA-Z\uac00-\ud7a3]+",
            "po": "[^_a-zA-Z\u0080-\u00ff]+"
        }
        self.fo_lang_limits = {
            "en": 3,
            "zh": 2,
            "jp": 2,
            "ko": 2,
            "fr": 3,
            "po": 3
        }
        self.cleanse_patterns = None
        self.token_pattern = "(?<=#) *?[0-9]+"
        self.linebreak_pattern = "[\n\t\r]+"
        self.space_pattern = " +"
        # camel_case
        self.camel_case_pattern = "(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
        # tokenizer
        self.zh_tok = hanlp.load("COARSE_ELECTRA_SMALL_ZH") if fo_lang_code == "zh" else None
        # stopwords
        self.stopwords: set
        self.__init_stopwords(project_name if use_project_stopwords else "")
        self.lang_dict = {
            "fr": "french",
            "ge": "german",
            "it": "italian"
        }
        # stemmer
        self.en_stemmer = self.get_stemmer("en")
        self.fo_stemmer = self.get_stemmer(fo_lang_code) if fo_lang_code is not None and fo_lang_code != "en" else None
        print("Preprocessor has inited in {:.1f}s.".format(time.time() - start_time))

    def __init_stopwords(self, project_name=""):
        self.stopwords = IOUtils.read_all_stopwords(stopwords_path)
        if project_name != "":
            self.stopwords = self.stopwords | IOUtils.read_stopwords(
                os.path.join(stopwords_path, "project", "stopwords_{}.txt".format(project_name)))

    def __init_cleanse_patterns(self):
        self.cleanse_patterns = [self.bilingual_lang_patterns.get(self.fo_lang_code)]

    def __remove_customer_stopwords(self, tokens: list[str]):
        return [token for token in tokens if token not in self.stopwords]

    def __remove_many_stopwords(self, tokens: list[str], lang_code="en"):
        if lang_code not in ("en", "zh", "fr"):
            return tokens
        stopwords = many_stop_words.get_stop_words(lang_code)
        return [token for token in tokens if token not in stopwords]

    def remove_stopwords(self, tokens: list[str]):
        tokens = self.__remove_customer_stopwords(tokens)
        if self.fo_lang_code is not None and self.fo_lang_code != "en":
            tokens = self.__remove_many_stopwords(tokens, self.fo_lang_code)
        return self.__remove_many_stopwords(tokens)

    def split_camel_case(self, camel_case_phrase: str):
        return re.split(self.camel_case_pattern, camel_case_phrase)

    def tokenize(self, text):
        tokens = text.split()
        bilingual_tokens = []
        for token in tokens:
            if self.fo_lang_code == "en" or token.encode("UTF-8").isalpha():
                bilingual_tokens.append(token)
            elif self.fo_lang_code == "zh":
                bilingual_tokens.extend(self.zh_tok(token))
            else:
                bilingual_tokens.append(token)
        return bilingual_tokens

    def get_stemmer(self, lang_code=None):
        language = self.lang_dict.get(lang_code, "english")
        return nltk.SnowballStemmer(language)

    def stemming(self, tokens: list[str]):
        tokens = [self.en_stemmer.stem(token) for token in tokens]
        if self.fo_stemmer is not None:
            tokens = [self.fo_stemmer.stem(token) for token in tokens]
        return tokens

    def preprocess(self, link_set) -> LinkSet:
        def cleanse_text(text: str) -> str:
            if not self.cleanse_patterns:
                self.__init_cleanse_patterns()
            cleansed_text = text
            for pattern in self.cleanse_patterns:
                cleansed_text = re.sub(pattern, " ", cleansed_text)
            cleansed_text = re.sub(self.fo_lang_patterns[self.fo_lang_code], r" \1 ", cleansed_text)
            cleansed_text = re.sub(self.linebreak_pattern, "\n", cleansed_text)
            cleansed_text = re.sub(self.space_pattern, " ", cleansed_text)
            return cleansed_text

        def cleanse_tokens(cleansed_text: str) -> list[str]:
            def limit_token_min_length(token_list):
                limited_tokens = []
                for a_token in token_list:
                    if a_token.encode("UTF-8").isalpha() and len(a_token) >= 3:
                        limited_tokens.append(a_token)
                    elif re.match(self.fo_lang_patterns[self.fo_lang_code], a_token) \
                            and len(a_token) >= self.fo_lang_limits[self.fo_lang_code]:
                        limited_tokens.append(a_token)
                return limited_tokens

            # camelcase split
            tokens = self.tokenize(cleansed_text)
            camel_tokens = []
            for token in tokens:
                camel_tokens.extend(self.split_camel_case(token))

            tokens = limit_token_min_length(camel_tokens)
            tokens = [token.lower() for token in tokens]

            tokens = self.remove_stopwords(tokens)
            tokens = self.stemming(tokens)
            tokens = self.remove_stopwords(tokens)

            return tokens

        print("Cleansing issues...")
        issue_start = time.time()
        for issue_id in tqdm(link_set.issues, file=sys.stdout, colour="#646cff"):
            issue_summary, issue_description = link_set.issues[issue_id]["summary"], \
                                               link_set.issues[issue_id]["description"]
            issue_summary, issue_description = cleanse_text(issue_summary), cleanse_text(issue_description)
            issue_tokens = cleanse_tokens(issue_summary + "\n" + issue_description)
            issue_tokens.append(issue_id)
            link_set.issues[issue_id]["tokens"] = " ".join(issue_tokens)
        print("Issues has been cleansed in {:.1f}s.".format(time.time() - issue_start))

        print("Cleansing commits...")
        commit_start = time.time()
        for commit_id in tqdm(link_set.commits, file=sys.stdout, colour="#646cff"):
            commit_message, commit_diff = link_set.commits[commit_id]["message"], link_set.commits[commit_id]["diff"]
            _tokens = [re.sub(" ", "", token) for token in re.findall(self.token_pattern, commit_message)]
            commit_message, commit_diff = cleanse_text(commit_message), cleanse_text(commit_diff)
            commit_tokens = cleanse_tokens(commit_message + "\n" + commit_diff)
            commit_tokens.extend(_tokens)
            link_set.commits[commit_id]["tokens"] = " ".join(commit_tokens)
        print("Commits has been cleansed in {:.1f}s.".format(time.time() - commit_start))

        return link_set
