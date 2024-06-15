class LinkSet:
    def __init__(self, issues: dict[str:str], commits: dict[str:str], links: list[tuple], source_name="issues",
                 target_name="commits"):
        self.issues = issues
        self.commits = commits
        self.gold_links = links
        self.source_name = source_name
        self.target_name = target_name

    def get_id(self):
        return self.source_name + "-" + self.target_name

    def get_source(self):
        if self.source_name == "issues":
            return self.issues
        return self.commits

    def get_source_size(self):
        if self.source_name == "issues":
            return len(self.issues)
        return len(self.commits)

    def get_target(self):
        if self.target_name == "commits":
            return self.commits
        return self.issues

    def get_target_size(self):
        if self.target_name == "commits":
            return len(self.commits)
        return len(self.issues)

    def get_all_docs(self) -> list[str]:
        all_docs = list(self.issues.values())
        all_docs.extend(list(self.commits.values()))
        return all_docs