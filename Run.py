import configparser
import os
import sys
import time

from IR.Experiment import Experiment

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini"))
dataset_path = config.get("path", "dataset_path")


def run_in_VSM(repo, version, fo_lang_code="zh", file_prefix=""):
    exp = Experiment(data_dir=dataset_path, model_type='vsm', fo_lang_code=fo_lang_code, repo_path=repo,
                     repo_version=version, file_prefix=file_prefix,
                     use_biterm=biterm_config)
    exp.do_preprocess()
    return exp.do_experiment()


def run_experiment(repo_list, version, fo_lang_code, prefix=""):
    results = {}
    for repo in repo_list:
        start_time = time.time()
        print("\033[31m" + repo + "\033[0m")
        sys.stdout = open(os.devnull, "w")
        results.setdefault(repo, run_in_VSM(repo=repo, version=version,
                                            fo_lang_code=fo_lang_code,
                                            file_prefix=prefix))
        sys.stdout = sys.__stdout__
        print("Experiment has done in {:.1f}s.".format(time.time() - start_time))
    for repo in results:
        print("{} {}".format(results[repo]["AP"], results[repo]["MAP"]))
    return results


if __name__ == "__main__":
    repos = [
        "arthas",
        # "awesome-berlin",
        # "bk-cmdb",
        # "canal",
        # "cica",
        # "druid",
        # "emmagee",
        # "konlpy",
        # "nacos",
        # "ncnn",
        # "pegasus",
        # "QMUI_Android",
        # "QMUI_IOS",
        # "rax",
        # "san",
        # "weui",
        # "xLua"
    ]

    # Configure Translation Version Path
    # It can be configured as "Multi_trans", "NLLB_trans", "M2M100_trans", "Tencent_trans" or "Google_trans".
    # It can also be configured as "", which means "No Translation"
    version_path = "Multi_trans"

    # Configure non-English languages: "zh"(chinese), "jp"(japanese), "po"(Portuguese), "ko"(Korean)
    # If artifacts have been translated, it can be configured as "en"(english)
    foreign = "en"

    pre = ""

    # "biterm_config" can be configured as "", "biterm" or "biterm score"
    # "": Basic VSM
    # "biterm": Basic VSM + Biterm
    # "biterm score": Basic VSM + AVIATE
    for biterm_config in ["biterm score"]:
        print("\033[31m" + biterm_config + "\033[0m")
        run_experiment(repo_list=repos, version=version_path, fo_lang_code=foreign, prefix=pre)
