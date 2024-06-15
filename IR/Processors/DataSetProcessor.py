import os
import IOUtils
from IR.DataSets.LinkSet import LinkSet


class DataSetProcessor:
    def __init__(self, data_dir, repo_path, repo_version):
        self.data_dir = os.path.join(data_dir, repo_path)
        self.dataset_dir = os.path.join(data_dir, repo_path, repo_version)

    def get_link_set_from_json(self, prefix="") -> LinkSet:
        links_path = os.path.join(self.data_dir, "links.json")
        issues_path = os.path.join(self.dataset_dir, prefix + "issues.json")
        commits_path = os.path.join(self.dataset_dir, prefix + "commits.json")
        return IOUtils.read_json_link_set(issues_path, commits_path, links_path)

    def save_link_set_to_json(self, link_set: LinkSet, prefix=""):
        IOUtils.save_as_json_file(self.dataset_dir, "issues.json", link_set.issues, prefix)
        IOUtils.save_as_json_file(self.dataset_dir, "commits.json", link_set.commits, prefix)
