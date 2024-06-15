import configparser
import os.path
import time

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from Translators.BaseTranslator import BaseTranslator

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config.ini"), encoding="UTF-8")

print("Loading NLLB......")
start = time.time()
nllb_path = config.get("path", "nllb_path")
tokenizer = AutoTokenizer.from_pretrained(nllb_path)
model = AutoModelForSeq2SeqLM.from_pretrained(nllb_path)
print("NLLB has been loaded in {:.1f}s.".format(time.time() - start))


class NLLBTranslator(BaseTranslator):
    def __init__(self, lang="zh"):
        super().__init__(lang)
        self.bilingual_translator = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            src_lang=self.bilingual_table[lang],
            tgt_lang="eng_Latn",
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
