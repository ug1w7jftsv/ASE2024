import json
import os
import time

from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.tmt.v20180321 import tmt_client, models

from Translators.BaseTranslator import BaseTranslator


class TencentTranslator(BaseTranslator):
    def __init__(self, lang="zh"):
        super().__init__(lang)
        # Need a credential of tencent translation
        self.cred = credential.Credential(
            os.environ.get("TENCENTCLOUD_SECRET_ID"),
            os.environ.get("TENCENTCLOUD_SECRET_KEY")
        )
        http_profile = HttpProfile()
        http_profile.endpoint = "tmt.tencentcloudapi.com"
        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile
        self.params = {
            "SourceTextList": [],
            "Source": lang,
            "Target": "en",
            "ProjectId": 0
        }
        self.bilingual_translator = tmt_client.TmtClient(self.cred, "ap-shanghai", client_profile)

    def call_translator(self, sentences):
        req = models.TextTranslateBatchRequest()
        self.params["SourceTextList"] = sentences
        req.from_json_string(json.dumps(self.params))
        return json.loads(self.bilingual_translator.TextTranslateBatch(req).to_json_string())["TargetTextList"]

    def translate_sentences(self, sentences: list[str], indexes: list[int]):
        sentences_to_translate = [sentences[index] for index in indexes]
        too_long_sentences = []
        for i in range(len(sentences_to_translate)):
            sentence = sentences_to_translate[i]
            if len(sentence) > 2000:
                too_long_sentences.append((i, sentence))
                sentences_to_translate[i] = ""

        sentences_to_translate_group = []
        a_group = []
        length = 0
        for sentence in sentences_to_translate:
            length += len(sentence)
            if length > 6000:
                sentences_to_translate_group.append(a_group)
                a_group = [sentence]
            else:
                a_group.append(sentence)
        if len(a_group) > 0:
            sentences_to_translate_group.append(a_group)

        sentences_to_translate = []
        for group in sentences_to_translate_group:
            group = self.call_translator(group)
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
