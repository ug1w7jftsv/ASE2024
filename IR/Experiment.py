import os
import time
import IOUtils
from collections import defaultdict

from IR.DataSets.DataSet import DataSet

from IR.Processors.DataSetProcessor import DataSetProcessor
from IR.Processors.Preprocessor import Preprocessor
from IR.Measures.Metrics import Metrics
from IR.IR_Model.VSM import VSM


class Experiment:
    def __init__(self, data_dir, model_type, fo_lang_code, repo_path, repo_version,
                 file_prefix="", link_threshold_interval=5, output_sub_dir="", use_biterm=""):
        self.data_dir = data_dir
        self.file_prefix = file_prefix
        self.model_type = model_type
        self.repo_path = repo_path
        self.repo_version = repo_version
        self.fo_lang_code = fo_lang_code
        self.link_threshold_interval = link_threshold_interval
        self.output_sub_dir = output_sub_dir

        # Configurations
        self.use_biterm = use_biterm.replace(" ", "_")
        self.use_biterm_word = "biterm" in use_biterm or "all" in use_biterm
        self.use_biterm_lambda = "lambda" in use_biterm or "all" in use_biterm
        self.use_biterm_theta = "theta" in use_biterm or "all" in use_biterm
        self.use_biterm_score = "score" in use_biterm or "all" in use_biterm
        self.lambda_weights = IOUtils.read_json_file(
            os.path.join(data_dir, repo_path, repo_version, file_prefix + "lambda.json")
        ) if self.use_biterm_lambda else None
        self.theta_weights = IOUtils.read_json_file(
            os.path.join(data_dir, repo_path, repo_version, file_prefix + "theta.json")
        ) if self.use_biterm_theta else None
        self.issues_scores = IOUtils.read_json_file(
            os.path.join(data_dir, repo_path, repo_version, file_prefix + "issues_score.json")
        ) if self.use_biterm_score else None
        self.commits_scores = IOUtils.read_json_file(
            os.path.join(data_dir, repo_path, repo_version, file_prefix + "commits_score.json")
        ) if self.use_biterm_score else None
        self.emphasis_scores = self.calculate_emphasis_scores() if self.use_biterm_score else None

        self.model = None
        self.metrics = None
        self.preprocessor = None

    def __build_model(self, docs: list[str]):
        model = None
        if self.model_type == "vsm":
            model = VSM(fo_lang_code=self.fo_lang_code)
            model.build_model(docs)
        self.model = model

    def get_links_with_similarity(self, dataset: DataSet) -> dict[str:list[tuple]]:
        generated_links_from_dataset = dict()
        for link_set_id in dataset.link_sets:
            link_set = dataset.link_sets[link_set_id]
            generated_links = self.model.generate_links_with_scores(
                link_set.get_source(), link_set.get_target()
            )
            generated_links_from_dataset[link_set_id] = generated_links
        return generated_links_from_dataset

    def _get_all_weight(self, _issue_id, _commit_id):
        lambda_weight = self.lambda_weights[_issue_id].get(_commit_id, 0)
        theta_weight = self.theta_weights[_issue_id].get(_commit_id, 0)
        return lambda_weight + theta_weight

    def _get_lambda_weight(self, _issue_id, _commit_id):
        lambda_weight = self.lambda_weights[_issue_id].get(_commit_id, 0)
        return lambda_weight

    def _get_theta_weight(self, _issue_id, _commit_id):
        theta_weight = self.theta_weights[_issue_id].get(_commit_id, 0)
        return theta_weight

    def calculate_emphasis_scores(self):
        emphasis_scores = defaultdict(float)
        avg_num = 0
        avg_sum = 0
        for issue_id in self.issues_scores:
            for key in self.issues_scores[issue_id]:
                avg_num += 1
                avg_sum += self.issues_scores[issue_id][key]
        for commit_id in self.commits_scores:
            for key in self.commits_scores[commit_id]:
                avg_num += 1
                avg_sum += self.commits_scores[commit_id][key]
        avg_sum = avg_sum / avg_num if avg_num != 0 else 0
        for issue_id in self.issues_scores:
            emphasis_scores[issue_id] = max(avg_sum, self.issues_scores[issue_id]["summary"]) * 0.1
        for commit_id in self.commits_scores:
            emphasis_scores[commit_id] = max(avg_sum, self.commits_scores[commit_id]["message"]) * 0.1
        return emphasis_scores

    def repetition(self, link_set):
        for issue_id in link_set.issues:
            issue_tokens = link_set.issues[issue_id]["tokens"].split(" ")
            if self.use_biterm_word:
                if self.emphasis_scores:
                    summary_biterms = link_set.issues[issue_id]["biterm"][0].split(" ")
                    if len(summary_biterms) > 0:
                        emphasis_score = self.emphasis_scores[issue_id]
                        repetition_count = round(len(issue_tokens) * min(emphasis_score, 1 / len(summary_biterms)))
                        for i in range(repetition_count):
                            issue_tokens.extend(summary_biterms)
                issue_tokens.extend(link_set.issues[issue_id]["biterm"])
            link_set.issues[issue_id] = " ".join(issue_tokens)

        for commit_id in link_set.commits:
            commit_tokens = link_set.commits[commit_id]["tokens"].split(" ")
            if self.use_biterm_word:
                if self.emphasis_scores:
                    message_biterms = link_set.commits[commit_id]["biterm"][0].split(" ")
                    if len(message_biterms) > 0:
                        emphasis_score = self.emphasis_scores[commit_id]
                        repetition_count = round(len(commit_tokens) * min(emphasis_score, 1 / len(message_biterms)))
                        for i in range(repetition_count):
                            commit_tokens.extend(message_biterms)
                commit_tokens.extend(link_set.commits[commit_id]["biterm"])
            link_set.commits[commit_id] = " ".join(commit_tokens)
        return link_set

    def do_preprocess(self):
        print("Preprocessing...")
        start_time = time.time()
        self.preprocessor = Preprocessor(self.fo_lang_code)
        dataset_processor = DataSetProcessor(self.data_dir, self.repo_path, self.repo_version)

        link_set = self.preprocessor.preprocess(dataset_processor.get_link_set_from_json(prefix=self.file_prefix))
        link_set = self.repetition(link_set)
        dataset_processor.save_link_set_to_json(link_set, prefix="_")
        print("Preprocess has done in {:.1f}s.".format(time.time() - start_time))

    def do_experiment(self):
        dataset_processor = DataSetProcessor(self.data_dir, self.repo_path, self.repo_version)
        dataset = DataSet([dataset_processor.get_link_set_from_json(prefix="_")])
        all_docs = dataset.get_all_docs()
        self.__build_model(all_docs)
        links_with_scores = self.get_links_with_similarity(dataset)

        result = {}
        for link_set_id in dataset.link_sets:
            print("Processing link set {}".format(link_set_id))
            if not self.metrics:
                self.metrics = Metrics(links_with_scores[link_set_id], dataset.link_sets[link_set_id].gold_links)
            else:
                self.metrics.set_measure_info(links_with_scores[link_set_id], dataset.link_sets[link_set_id].gold_links)
            sorted_link_set = self.metrics.sorted_link_set
            _ap = self.metrics.calculate_AP()
            _map = self.metrics.calculate_MAP()

            # precision,recall,F1
            threshold = 0
            scores = []
            while threshold <= 100:
                filter_links_above_threshold = [x for x in sorted_link_set if x[2] >= threshold / 100]
                eval_score = dataset.evaluate_link_set(link_set_id, filter_links_above_threshold)
                scores.append(eval_score)
                threshold += self.link_threshold_interval

            link_set_info = "{} issues, {} commits and {} links remains after preprocessing.".format(
                len(dataset.link_sets[link_set_id].issues), len(dataset.link_sets[link_set_id].commits),
                len(dataset.link_sets[link_set_id].gold_links))
            print(link_set_info)
            print("AP = {}".format(_ap))
            print("MAP = {}".format(_map))
            print("Precision, Recall, F1")
            print(scores)

            result = {"AP": _ap, "MAP": _map}
        return result

    def save_experiment(self, link_set_id, ap, _map, scores, link_set, link_sets_with_key, link_set_info):
        trans_postfix = self.file_prefix[:-1] if self.file_prefix != "" else "origin"
        save_dir = os.path.join("results", self.output_sub_dir, self.repo_path,
                                "_".join([self.model_type, trans_postfix]))
        file_name = "measured.txt"
        link_score_name = "links_score.json"
        link_with_key_score_name = "links_with_key_score.json"

        measure_info = [link_set_info, "AP = {}".format(ap), "MAP = {}".format(_map), "Precision, Recall, F1",
                        str(scores)]
        IOUtils.save_as_text_file(save_dir, file_name, measure_info, link_set_id)
        IOUtils.save_as_json_file(save_dir, link_score_name, link_set, link_set_id)
        IOUtils.save_as_json_file(save_dir, link_with_key_score_name, link_sets_with_key, link_set_id)
