import configparser
import os.path
import time
from collections import defaultdict

import IOUtils

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config.ini"))


class Metrics:
    def __init__(self, links_sets: dict[str:list[tuple]] = None, gold_links: list[tuple] = None, round_digit_num=4):
        # dict[issue_id:list[link_with_score]]
        self.sorted_link_sets_with_key = None
        # list[link_with_score]
        self.sorted_link_set = None
        # dict[issue_id:set(link)]
        self.gold_links_with_key = None
        # set(link)
        self.gold_links = None
        # dict[issue_id:dict[rank:(precision,recall)]], for calculate MAP
        self.measure_table_with_key = None
        self.hit_table_with_key = None
        # dict[rank:(precision,recall)], for calculate AP
        self.measure_table = None
        self.hit_table = None
        # init measure table
        self.set_measure_info(links_sets, gold_links)
        self.round_digit_num = round_digit_num

    def set_measure_info(self, links_sets: dict[str:list[tuple]] = None, gold_links: list[tuple] = None):
        start_time = time.time()
        self.gold_links_with_key = defaultdict(set)
        if links_sets is not None:
            self.sorted_link_sets_with_key = {key: sorted(links_sets[key], key=lambda k: k[2], reverse=True) for key in
                                              links_sets}
            sorted_links = []
            for link_set_value in links_sets.values():
                sorted_links.extend(link_set_value)
            self.sorted_link_set = sorted(sorted_links, key=lambda k: k[2], reverse=True)
        if gold_links is not None:
            self.gold_links = set(gold_links)
            for gold_link in gold_links:
                self.gold_links_with_key[gold_link[0]].add(gold_link)
        print("Measure information has been sorted in {:.4f}s.".format(time.time() - start_time))
        self.__generate_measure_table()

    # Precision and Recall table
    def __generate_measure_table(self):
        start_time = time.time()
        self.measure_table_with_key = defaultdict(dict)
        self.hit_table_with_key = defaultdict(list)
        self.measure_table = dict()
        self.hit_table = []
        # 生成按issue_id分类的度量表
        for source_key in self.sorted_link_sets_with_key:
            hit_num = 0
            link_set = self.sorted_link_sets_with_key[source_key]
            gold_links = self.gold_links_with_key[source_key]
            for i in range(len(link_set)):
                issue_id, commit_id, score = link_set[i]
                if (issue_id, commit_id) in gold_links:
                    hit_num += 1
                    self.hit_table_with_key[source_key].append(i)
                precision = hit_num / (i + 1)
                recall = hit_num / len(gold_links)
                self.measure_table_with_key[source_key][i] = (precision, recall)
        hit_num = 0
        link_set = self.sorted_link_set
        gold_links = self.gold_links
        for i in range(len(link_set)):
            issue_id, commit_id, score = link_set[i]
            if (issue_id, commit_id) in gold_links:
                hit_num += 1
                self.hit_table.append(i)
            precision = hit_num / (i + 1)
            recall = hit_num / len(gold_links)
            self.measure_table[i] = (precision, recall)
        print("Measure table has been built in {:.4f}s.".format(time.time() - start_time))

    # Precision
    def get_precision(self, index, source_key=None):
        return self.measure_table[index][0] if not source_key else self.measure_table_with_key[source_key][index][0]

    # Recall
    def get_recall(self, index, source_key=None):
        return self.measure_table[index][1] if not source_key else self.measure_table_with_key[source_key][index][1]

    # Precision, Recall
    def get_metrics(self, index, source_key=None):
        return self.measure_table[index] if not source_key else self.measure_table_with_key[source_key][index]

    # Average Precision
    def calculate_AP(self, source_key=None):
        hit_table = self.hit_table if not source_key else self.hit_table_with_key[source_key]
        precision_sum = 0
        for hit_index in hit_table:
            precision_sum += self.get_precision(hit_index, source_key=source_key)
        return round(precision_sum / len(self.gold_links if not source_key else self.gold_links_with_key[source_key]),
                     self.round_digit_num)

    # Mean Average Precision
    def calculate_MAP(self):
        ap_sum = 0
        for source_key in self.sorted_link_sets_with_key:
            ap = self.calculate_AP(source_key=source_key)
            ap_sum += ap
        return round(ap_sum / len(self.sorted_link_sets_with_key), self.round_digit_num)

    def save_measure_table(self, repo, version="", *args):
        IOUtils.save_as_json_file(save_dir=os.path.join(config.get("path", "results_path"), repo),
                                  file_name="_".join(["PR", version.replace("\\", "_"), *args]) + ".json",
                                  obj=self.measure_table)
