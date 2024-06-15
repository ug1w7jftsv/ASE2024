import configparser
import os
import time

from transformers import pipeline, M2M100Tokenizer, M2M100ForConditionalGeneration

from Translators.BaseTranslator import BaseTranslator

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config.ini"), encoding="UTF-8")

print("Loading M2M100......")
start = time.time()
m2m100_path = config.get("path", "m2m100_path")
tokenizer = M2M100Tokenizer.from_pretrained(m2m100_path)
model = M2M100ForConditionalGeneration.from_pretrained(m2m100_path)
print("M2M100 has been loaded in {:.1f}s.".format(time.time() - start))


class M2M100Translator(BaseTranslator):
    def __init__(self, lang="zh"):
        super().__init__(lang)
        self.bilingual_translator = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            src_lang="zh",
            tgt_lang="en",
            max_length=512,
            device="cuda:0"
        )

    def translate_sentences(self, sentences: list[str], indexes: list[int]):
        sentences_to_translate = [sentences[index] for index in indexes]
        too_long_sentences = []
        for i in range(len(sentences_to_translate)):
            sentence = sentences_to_translate[i]
            if len(sentence) > 1024:
                too_long_sentences.append((i, sentence))
                sentences_to_translate[i] = ""
        sentences_to_translate = list(
            map(lambda x: x["translation_text"], self.bilingual_translator(sentences_to_translate))
        )
        if len(sentences_to_translate) > 0:
            print(sentences_to_translate)
        pointer = 0
        for index in indexes:
            sentences[index] = sentences_to_translate[pointer]
            pointer += 1
        for sentence_info in too_long_sentences:
            sentences[sentence_info[0]] = sentence_info[1]
        return sentences
