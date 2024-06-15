import configparser
import copy
import math
import os.path
import re
import sys
import time
from collections import defaultdict

import nltk
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm

import IOUtils

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config.ini"), encoding="UTF-8")

nlp_pipline = StanfordCoreNLP(config.get("path", "stanfordcorenlp_path"))
stopwords = IOUtils.read_all_stopwords("../stopwords")
en_stemmer = nltk.SnowballStemmer("english")
exclude_relation_table = {"ROOT", "punct", "aux", "det", "mark"}
legal_pos_table = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}


def pruning(issues_biterm, commits_biterm, consensus):
    for issue_id in issues_biterm:
        for key in issues_biterm[issue_id]:
            issues_biterm[issue_id][key] = list(
                filter(lambda x: tuple(x) in consensus, issues_biterm[issue_id][key]))
    for commit_id in commits_biterm:
        for key in commits_biterm[commit_id]:
            commits_biterm[commit_id][key] = list(
                filter(lambda x: tuple(x) in consensus, commits_biterm[commit_id][key]))


# Merge biterms into files
def merging(issues, commits, issues_biterm, commits_biterm):
    for issue_id in issues:
        biterm_list = []
        # key = summary/description
        for key in issues_biterm[issue_id]:
            biterm_list.append(list(map(lambda x: "".join(sorted([x[0], x[1]])), issues_biterm[issue_id][key])))
        issues[issue_id]["biterm"] = list(map(lambda x: " ".join(x), biterm_list))
    for commit_id in commits:
        biterm_list = []
        # key = message/diff
        for key in commits_biterm[commit_id]:
            biterm_list.append(list(map(lambda x: "".join(sorted([x[0], x[1]])), commits_biterm[commit_id][key])))
        commits[commit_id]["biterm"] = list(map(lambda x: " ".join(x), biterm_list))


def preprocess(term):
    term = [t.lower() for t in
            filter(lambda x: len(x) >= 3, re.split("(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", term))]
    term = [en_stemmer.stem(t) for t in filter(lambda x: x not in stopwords, term)]
    term = list(filter(lambda x: x not in stopwords, term))
    return " ".join(term)


def get_doc_text_list(text_dict: dict):
    if "summary" in text_dict:
        return [text_dict["summary"], text_dict["description"]], "summary", "description"
    elif "message" in text_dict:
        return [text_dict["message"], text_dict["diff"]], "message", "diff"


def get_sentences_and_variants(key, info, token_term_map, sentences=None, indexes=None):
    if sentences is None:
        sentences = re.split("\n", info[key])
    if indexes is None:
        indexes = info[key + "_bilingual_indexes"]
    # multi_trans::list<list<translation variants>>
    multi_trans = info[key + "_multi_trans"]
    pointer = 0
    for index in indexes:
        # delete english sentences
        sentences[index] = ""
        pointer += 1
    return re.split("\n+", "\n".join(sentences)), [
        [extract_from_sentences([sentence], token_term_map) for sentence in variants] for variants in
        multi_trans]


def calculate_score(issues_sentences_biterms, commits_sentences_biterms,
                    issues_multi_biterms, commits_multi_biterms, consensual_biterms):
    ITVF_cache = {}

    def __cal_ITVF(biterm):
        if not ITVF_cache.get(biterm):
            m = 0
            for sent in all_sentences_biterms_dict:
                if biterm in sent:
                    m += 1
            ITVF_cache[biterm] = math.log(n / m, 10) if m > 0 else 0
        return ITVF_cache[biterm]

    def __cal_variants_biterm_ITVF(variants_dict, mode="variants"):
        variants_biterm_itvf_dict = defaultdict(dict)
        for _doc_id in variants_dict:
            for _key in variants_dict[_doc_id]:
                variants_biterm_itvf_dict[_doc_id][_key] = []
                for variants_group in variants_dict[_doc_id][_key]:
                    scores = []
                    if mode == "variants":
                        for variant in variants_group:
                            v_itvf = [__cal_ITVF(biterm) for biterm in variant]
                            scores.append(v_itvf)
                            all_score.extend(v_itvf)
                    else:
                        for biterm in variants_group:
                            scores.append(__cal_ITVF(biterm))
                        all_score.extend(scores)
                    variants_biterm_itvf_dict[_doc_id][_key].append(scores)
        return variants_biterm_itvf_dict

    def __minmax(values, min_value, max_value):
        res = []
        for value in values:
            if value > max_value:
                res.append(1)
            elif value < min_value:
                res.append(0)
            else:
                res.append((value - min_value) / (max_value - min_value))
        return res

    def __cal_minmax_ITVF_TVF(score_dict, mode="variants"):
        for _doc_id in score_dict:
            for _key in score_dict[_doc_id]:
                for index in range(len(score_dict[_doc_id][_key])):
                    if mode == "variants":
                        score_dict[_doc_id][_key][index] = [__minmax(s, max_value=maximum, min_value=minimal)
                                                            for s in score_dict[_doc_id][_key][index]]
                        all_score.extend(score_dict[_doc_id][_key][index])
                    else:
                        score_dict[_doc_id][_key][index] = __cal_average(__minmax(
                            score_dict[_doc_id][_key][index], max_value=maximum, min_value=minimal))
                        all_score.append(score_dict[_doc_id][_key][index])
        return score_dict

    # Build sentences dictionary and calculate N.
    all_sentences_biterms_dict = []
    for sent_biterms in [issues_sentences_biterms, commits_sentences_biterms]:
        for doc_id in sent_biterms:
            for key in sent_biterms[doc_id]:
                all_sentences_biterms_dict.extend(sent_biterms[doc_id][key])
    for multi_biterms in [issues_multi_biterms, commits_multi_biterms]:
        for doc_id in multi_biterms:
            for key in multi_biterms[doc_id]:
                for sentences in multi_biterms[doc_id][key]:
                    all_sentences_biterms_dict.extend(sentences)
    n = len(all_sentences_biterms_dict)

    all_score = []
    issues_sentences_tvf_itfv = __cal_variants_biterm_ITVF(issues_sentences_biterms, mode="sentences")
    commits_sentences_tvf_itfv = __cal_variants_biterm_ITVF(commits_sentences_biterms, mode="sentences")
    issues_variants_tvf_itfv = __cal_variants_biterm_ITVF(issues_multi_biterms)
    commits_variants_tvf_itfv = __cal_variants_biterm_ITVF(commits_multi_biterms)

    # Standardization
    avg = __cal_average(all_score)
    std = __cal_std(all_score)
    maximum = max(filter(lambda x: x <= avg + 3 * std, all_score))
    minimal = min(filter(lambda x: x >= avg - 3 * std, all_score))

    all_score = []
    issues_sentences_tvf_itfv = __cal_minmax_ITVF_TVF(issues_sentences_tvf_itfv, mode="sentences")
    commits_sentences_tvf_itfv = __cal_minmax_ITVF_TVF(commits_sentences_tvf_itfv, mode="sentences")
    issues_variants_tvf_itfv = __cal_minmax_ITVF_TVF(issues_variants_tvf_itfv)
    commits_variants_tvf_itfv = __cal_minmax_ITVF_TVF(commits_variants_tvf_itfv)
    return issues_sentences_tvf_itfv, commits_sentences_tvf_itfv, issues_variants_tvf_itfv, commits_variants_tvf_itfv


def __cal_average(arr):
    return sum(arr) / len(arr) if len(arr) > 0 else 0


def __cal_std(arr):
    average = __cal_average(arr)
    variance_arr = [(num - average) ** 2 for num in arr]
    return math.sqrt(sum(variance_arr) / len(arr)) if len(arr) > 0 else 0


def __select_biterms(sentences_s_dict, variants_s_dict, variants_biterms, variants_multi_biterms):
    select_indexes = defaultdict(dict)
    select_scores = sentences_s_dict
    for _doc_id in variants_s_dict:
        for _key in variants_s_dict[_doc_id]:
            select_indexes[_doc_id][_key] = []
            for variants_index in range(len(variants_s_dict[_doc_id][_key])):
                variants = variants_multi_biterms[_doc_id][_key][variants_index]
                variants_bt_tvf_itvf = variants_s_dict[_doc_id][_key][variants_index]
                variants_tvf_itvf = [__cal_average(variant_tvf_itvf) for variant_tvf_itvf in variants_bt_tvf_itvf]

                max_s = max(variants_tvf_itvf)
                max_s_index = variants_tvf_itvf.index(max_s)
                select_scores[_doc_id][_key].append(max_s)
                select_indexes[_doc_id][_key].append(max_s_index)
                if max_s != 0:
                    threshold = min([0.6, max_s])
                    selected_biterms = copy.deepcopy(variants[max_s_index])
                    for index in range(len(variants)):
                        for bt_index in range(len(variants[index])):
                            if variants[index][bt_index] not in selected_biterms \
                                    and variants_bt_tvf_itvf[index][bt_index] >= threshold:
                                selected_biterms.append(variants[index][bt_index])
                    variants_biterms[_doc_id][_key].extend(selected_biterms)
            select_scores[_doc_id][_key] = list(filter(lambda x: x > 0, select_scores[_doc_id][_key]))
            select_scores[_doc_id][_key] = __cal_average(select_scores[_doc_id][_key])
    return select_indexes, select_scores


def extract_from_sentences(sentences, token_term_map):
    biterm_list = []
    for sentence in sentences:
        sentence = re.sub("(?<=[a-zA-Z])[_\\-](?=[a-zA-Z])", " ", sentence)
        try:
            tokens = nlp_pipline.word_tokenize(sentence)
            for token in tokens:
                if token.encode("UTF-8").isalpha():
                    term = preprocess(token)
                    if term != "":
                        token_term_map[token] = term
            relations = nlp_pipline.dependency_parse(sentence)
            pos_tags = nlp_pipline.pos_tag(sentence)
            sentence_biterm_list = []
            for relation in relations:
                if relation[0] not in exclude_relation_table:
                    source_term, source_pos = pos_tags[relation[1] - 1]
                    target_term, target_pos = pos_tags[relation[2] - 1]
                    if source_pos in legal_pos_table and target_pos in legal_pos_table:
                        if source_term in token_term_map and target_term in token_term_map:
                            if token_term_map.get(source_term) != token_term_map.get(target_term):
                                sentence_biterm_list.append(
                                    (token_term_map.get(source_term), token_term_map.get(target_term),
                                     relation[0]))
            biterm_list.extend(sentence_biterm_list)
        except Exception:
            print("Exception!")
            continue
    return biterm_list


# biterm_dict::<id:{key:biterm}>
def extract_from_docs(doc_dict: dict[str, dict]):
    # <id:{summary/message:biterms,description/diff:biterms}>
    biterm_dict = defaultdict(dict)
    multi_biterm_dict = defaultdict(dict)
    token_term_map = dict()
    for doc_id in tqdm(doc_dict, file=sys.stdout, colour="#646cff"):
        text_list, key1, key2 = get_doc_text_list(doc_dict[doc_id])
        keys = [key1, key2]
        for i in range(2):
            # text::issue->summary/description; commit->message/diff
            sentences, multi_trans_biterms = get_sentences_and_variants(keys[i], doc_dict[doc_id],
                                                                        token_term_map)
            multi_biterm_dict[doc_id].setdefault(keys[i], multi_trans_biterms)
            biterm_list = [extract_from_sentences([sentence], token_term_map) for sentence in sentences]
            biterm_dict[doc_id].setdefault(keys[i], biterm_list)
    return biterm_dict, multi_biterm_dict


def merge_biterm_dict(biterm_dict1, biterm_dict2):
    merge_dict = copy.deepcopy(biterm_dict1)
    for doc_id in biterm_dict2:
        for key in biterm_dict2[doc_id]:
            for _variant_biterms in biterm_dict2[doc_id][key]:
                for _variant_biterm in _variant_biterms:
                    merge_dict[doc_id][key].extend(_variant_biterm)
    return merge_dict


# biterm_dict::<id:{key:biterm_dict<key:biterms>}>
def deduplicate(biterm_dict: dict[str, dict]):
    return {
        doc_id: {
            key: set(map(lambda x: tuple(x), biterm_dict[doc_id][key])) for key in biterm_dict[doc_id]
        } for doc_id in biterm_dict
    }


def tuplize(docs_biterms, mode="variants"):
    if mode == "variants":
        biterms = {
            _doc_id: {
                _key: list(map(lambda x: list(map(lambda y: list(map(lambda z: tuple(z), y)), x))
                               , docs_biterms[_doc_id][_key])) for _key in docs_biterms[_doc_id]
            } for _doc_id in docs_biterms
        }
        return biterms
    else:
        return {
            _doc_id: {
                _key: list(map(lambda x: list(map(lambda y: tuple(y), x)), docs_biterms[_doc_id][_key])) for
                _key in docs_biterms[_doc_id]
            } for _doc_id in docs_biterms
        }


# biterm_set::<id:{key:biterm_dict<key:biterm_set>}>
def get_all_biterm_set(biterm_set: dict[str, dict]):
    all_biterm_set = set()
    for doc_id in biterm_set:
        for key in biterm_set[doc_id]:
            all_biterm_set = all_biterm_set | biterm_set[doc_id][key]
    return all_biterm_set


def extract_biterm(issue_path, commit_path, save_dir, prefix="", use_cache=False):
    def extract_from_documents(doc_name, docs):
        print("Extract biterms from {}...".format(doc_name))
        if use_cache:
            sentences_biterms = IOUtils.read_json_file(os.path.join(save_dir, doc_name + "_biterm.json"))
            docs_multi_biterms = IOUtils.read_json_file(os.path.join(save_dir, doc_name + "_multi_biterms.json"))
            sentences_biterms = tuplize(sentences_biterms, mode="sentences")
            docs_multi_biterms = tuplize(docs_multi_biterms)
        else:
            # docs_biterms::<doc_id:list<biterm>> docs_multi_biterms::<doc_id:list<list<biterm>>>
            sentences_biterms, docs_multi_biterms = extract_from_docs(docs)
            IOUtils.save_as_json_file(save_dir, doc_name + "_biterm.json", sentences_biterms, prefix)
            IOUtils.save_as_json_file(save_dir, doc_name + "_multi_biterms.json", docs_multi_biterms, prefix)
        docs_biterms = {
            _doc_id: {
                _key: [bt for sent in sentences_biterms[_doc_id][_key] for bt in sent] for _key in
                sentences_biterms[_doc_id]
            } for _doc_id in sentences_biterms
        }
        all_doc_biterm = merge_biterm_dict(docs_biterms, docs_multi_biterms)
        docs_biterm_sets = deduplicate(all_doc_biterm)
        all_docs_biterm_set = get_all_biterm_set(docs_biterm_sets)
        return sentences_biterms, docs_biterms, docs_multi_biterms, all_doc_biterm, all_docs_biterm_set

    issues = IOUtils.read_json_file(issue_path)
    # issues_biterms::<issue_id:list<biterm>> issues_multi_biterms::<issue_id:list<list<biterm>>>
    issues_sentences_biterms, issues_biterms, issues_multi_biterms, all_issue_biterm, all_issues_biterm_set = \
        extract_from_documents("issues", issues)

    commits = IOUtils.read_json_file(commit_path)
    # commits_biterms::<issue_id:list<biterm>> commits_multi_biterms::<issue_id:list<list<biterm>>>
    commits_sentences_biterms, commits_biterms, commits_multi_biterms, all_commit_biterm, all_commits_biterm_set = \
        extract_from_documents("commits", commits)

    all_biterm_set = all_issues_biterm_set & all_commits_biterm_set
    all_biterm_dict = defaultdict(int)
    for biterms in [all_issue_biterm, all_commit_biterm]:
        for doc_id in biterms:
            for key in biterms[doc_id]:
                for biterm in biterms[doc_id][key]:
                    if biterm in all_biterm_set:
                        all_biterm_dict[biterm] += 1

    print("Calculating ConsDistinctiveness scores...")
    issues_sentences_score, commits_sentences_score, issues_variants_score, commits_variants_score = \
        calculate_score(issues_sentences_biterms,
                        commits_sentences_biterms,
                        issues_multi_biterms, commits_multi_biterms,
                        all_biterm_dict)

    print("Selecting biterms...")
    issues_selected, issues_scores = __select_biterms(issues_sentences_score, issues_variants_score, issues_biterms,
                                                      issues_multi_biterms)
    commits_selected, commits_scores = __select_biterms(commits_sentences_score, commits_variants_score,
                                                        commits_biterms, commits_multi_biterms)

    all_issues_biterm_set = get_all_biterm_set(deduplicate(issues_biterms))
    all_commits_biterm_set = get_all_biterm_set(deduplicate(commits_biterms))
    IOUtils.save_as_json_file(save_dir, "issues_score.json", issues_scores, prefix)
    IOUtils.save_as_json_file(save_dir, "commits_score.json", commits_scores, prefix)

    print("Pruning...")
    pruning(issues_biterms, commits_biterms, all_issues_biterm_set & all_commits_biterm_set)

    merge_and_save_data(save_dir, issues, commits, issues_biterms, commits_biterms, prefix)


def get_biterms_from_file(save_dir, prefix):
    issues_biterm = IOUtils.read_json_file(os.path.join(save_dir, prefix + "issues_biterm.json"))
    commits_biterm = IOUtils.read_json_file(os.path.join(save_dir, prefix + "commits_biterm.json"))
    return tuplize(issues_biterm, mode="sentences"), tuplize(commits_biterm, mode="sentences")


def merge_and_save_data(save_dir, issues=None, commits=None, issues_biterm=None, commits_biterm=None, prefix=""):
    if type(issues) is str:
        issues = IOUtils.read_json_file(issues)
    if type(commits) is str:
        commits = IOUtils.read_json_file(commits)
    if issues_biterm is None or commits_biterm is None:
        issues_biterm, commits_biterm = get_biterms_from_file(save_dir, prefix)
        print("Pruning...")
        pruning(issues_biterm, commits_biterm,
                get_all_biterm_set(deduplicate(issues_biterm)) & get_all_biterm_set(deduplicate(commits_biterm)))
    print("Merging...")
    merging(issues, commits, issues_biterm, commits_biterm)

    print("Saving...")
    IOUtils.save_as_json_file(save_dir, "issues.json", issues, prefix)
    IOUtils.save_as_json_file(save_dir, "commits.json", commits, prefix)


if __name__ == "__main__":
    dataset_base_dir = config.get("path", "dataset_path")
    save_base_dir = dataset_base_dir
    repo_list = [
        "arthas",
        "awesome-berlin",
        "bk-cmdb",
        "canal",
        "cica",
        "druid",
        "emmagee",
        "konlpy",
        "nacos",
        "ncnn",
        "pegasus",
        "QMUI_Android",
        "QMUI_IOS",
        "rax",
        "san",
        "weui",
        "xLua"
    ]
    prefix = ""
    # src: The path of the issues and commits
    src = "Multi_trans"
    # tgt: The path to save the biterms
    tgt = "Multi_trans"
    for repo in repo_list:
        print("\033[31m" + repo + "\033[0m")
        st = time.time()
        extract_biterm(
            os.path.join(dataset_base_dir, repo, src, prefix + "issues.json"),
            os.path.join(dataset_base_dir, repo, src, prefix + "commits.json"),
            os.path.join(save_base_dir, repo, tgt),
            prefix=prefix,
            use_cache=False
        )
        print("\033[31mExtraction has been done in {:.1f}s.\033[0m".format(time.time() - st))
    nlp_pipline.close()
