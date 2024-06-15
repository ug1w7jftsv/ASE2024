import configparser
import os
import time

import google.api_core.exceptions
from google.cloud import translate_v2 as translate
from Translators.BaseTranslator import BaseTranslator

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config.ini"), encoding="UTF-8")
auth_info = configparser.ConfigParser()
auth_info.read("auth.ini", encoding="UTF-8")  # Need a Google-API-Key.


class GoogleTranslator(BaseTranslator):
    def __init__(self, lang="zh"):
        super().__init__(lang)
        self.bilingual_translator = translate.Client()

    def translate_sentences(self, sentences: list[str], indexes: list[int]):
        sentences_to_translate = [sentences[index] for index in indexes]
        too_long_sentences = []
        for i in range(len(sentences_to_translate)):
            sentence = sentences_to_translate[i]
            if len(sentence) > 2000 or len(sentence) < 1:
                too_long_sentences.append((i, sentence))
                sentences_to_translate[i] = ""

        sentences_to_translate_group = []
        a_group = []
        length = 0
        for sentence in sentences_to_translate:
            length += len(sentence)
            if length > 6000 or len(a_group) >= 100:
                sentences_to_translate_group.append(a_group)
                a_group = [sentence]
            else:
                a_group.append(sentence)
        if len(a_group) > 0:
            sentences_to_translate_group.append(a_group)

        sentences_to_translate = []
        group_index = 0
        while group_index < len(sentences_to_translate_group):
            group = sentences_to_translate_group[group_index]
            try:
                group = list(map(lambda x: x["translatedText"], self.bilingual_translator.translate(group)))
                group_index += 1
            except google.api_core.exceptions.ServiceUnavailable:
                print("Reconnecting...")
                time.sleep(1)
                continue
            sentences_to_translate.extend(group)
            time.sleep(0.2)

        if len(sentences_to_translate) > 0:
            print(sentences_to_translate)
        pointer = 0
        for index in indexes:
            sentences[index] = sentences_to_translate[pointer]
            pointer += 1
        for sentence_info in too_long_sentences:
            sentences[sentence_info[0]] = sentence_info[1]
        return sentences
