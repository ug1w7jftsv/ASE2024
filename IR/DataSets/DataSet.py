from IR.DataSets.LinkSet import LinkSet


# DataSet contains many LinkSets, but there's only one here.
class DataSet:
    def __init__(self, link_sets: list[LinkSet], round_digit_num=4):
        self.link_sets = dict()
        for link_set in link_sets:
            self.link_sets[link_set.get_id()] = link_set
        self.round_digit_num = round_digit_num

    # get documents of link_set
    def get_all_docs(self) -> list[str]:
        docs = []
        for link_set_id in self.link_sets:
            docs.extend(self.link_sets.get(link_set_id).get_all_docs())
        return docs

    def evaluate_link_set(self, gold_link_set_id: str, eval_link_set):
        eval_links = set([(link[0], link[1]) for link in eval_link_set])
        gold_links = set(self.link_sets[gold_link_set_id].gold_links)
        # size = len(gold_link_set.issues) * len(gold_link_set.commits)
        tp = len(eval_links & gold_links)
        fp = len(eval_links - gold_links)
        fn = len(gold_links - eval_links)
        # tn = size - len(eval_links | gold_links)

        precision = tp / (tp + fp) if tp != 0 else 0
        recall = tp / (tp + fn) if tp != 0 else 0
        f1 = 2 * (recall * precision) / (recall + precision) if recall + precision != 0 else 0

        return round(precision, self.round_digit_num), round(recall, self.round_digit_num), round(f1,
                                                                                                  self.round_digit_num)

    def __str__(self):
        info = []
        for link_set_id in self.link_sets:
            link_set = self.link_sets[link_set_id]
            info.append(
                "Link_set_id:{}, Size:{}, Source_size:{}, Target_size:{}."
                .format(link_set_id, len(link_set.gold_links), link_set.get_source_size(), link_set.get_target_size())
            )
        return "\n".join(info)
