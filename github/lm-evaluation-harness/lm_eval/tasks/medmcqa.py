# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.

TODO: Write a Short Description of the task.

Homepage: TODO: Add the URL to the task's Homepage here.
"""
import numpy as np
from lm_eval.base import MultipleChoiceTask, rf


# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""


# TODO: Replace `NewTask` with the name of your Task.
class MedMCQA(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "medmcqa"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["validation"])  # ground truth is not available for test set

    def _process_doc(self, doc):
        out_doc = {
            "query": doc["question"],
            "choices": [doc['opa'], doc['opb'], doc['opc'], doc['opd']],
            "gold": doc['cop'],
        }
        return out_doc

    # def doc_to_text(self, doc):
    #     "Only question. Ref: OpenbookQA"
    #     return doc["query"]

    def doc_to_text(self, doc):
        return "Question: {}\nAnswer:".format(doc["query"])

    # def doc_to_text(self, doc):
    #     template = (
    #         "The following is a multiple-choice question (with options) about medical knowledge. "
    #         "Please select the most appropriate one from the following answer candidates as the final answer.\n\n"
    #         "### Question: {question}\n\n"
    #         "### Options:\n{options}\n\n"
    #         "### Answer:"
    #     )
    #     return template.format(
    #         question=doc["query"],
    #         options="\n".join(f"({'ABCDE'[i]}) {option}" for i, option in enumerate(doc["choices"])),
    #     )
    #
    # def construct_requests(self, doc, ctx):
    #     lls = [
    #         rf.loglikelihood(ctx, f"{choice}")[0] for choice in doc["choices"]
    #     ]
    #     lls.extend(
    #         [
    #             rf.loglikelihood(ctx, f" {choice}")[0] for choice in doc["choices"]
    #         ]
    #     )
    #     lls.extend(
    #         [
    #             rf.loglikelihood(ctx, f"({choice_id}) {choice}")[0] for choice_id, choice in zip('ABCD', doc["choices"])
    #         ]
    #     )
    #
    #     return lls
    #
    # def process_results(self, doc, results):
    #     gold = doc["gold"]
    #
    #     acc = 1.0 if np.argmax(results) % 4 == gold else 0.0
    #     completion_len = np.array([float(len(i)) for i in doc["choices"]] * 3)
    #     acc_norm = 1.0 if np.argmax(results / completion_len) % 4 == gold else 0.0
    #
    #     return {
    #         "acc": acc,
    #         "acc_norm": acc_norm,
    #     }

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]
