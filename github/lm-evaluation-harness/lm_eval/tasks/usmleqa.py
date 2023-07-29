# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.

TODO: Write a Short Description of the task.

Homepage: TODO: Add the URL to the task's Homepage here.
"""
from lm_eval.base import MultipleChoiceTask, rf


# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""


# TODO: Replace `NewTask` with the name of your Task.
class USMLEQA(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "GBaker/MedQA-USMLE-4-options"       # 4 options version
    DATASET_NAME = None

    # DATASET_PATH = "bigbio/med_qa"                    # 5 options version
    # DATASET_NAME = "med_qa_en_bigbio_qa"

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
        if self.DATASET_PATH == "bigbio/med_qa":
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        if self.DATASET_PATH == "bigbio/med_qa":
            # all_option_ids = [option['key'] for option in doc['options']]
            # all_options = [option['value'] for option in doc['options']]
            all_option_ids = ['A', 'B', 'C', 'D', 'E']
            all_options = doc['choices']

            out_doc = {
                "query": doc["question"],
                "choices": all_options,
                "gold": all_options.index(doc["answer"][0]),
            }

        elif self.DATASET_PATH == "GBaker/MedQA-USMLE-4-options":
            all_option_ids = ['A', 'B', 'C', 'D']
            all_options = [doc['options'][option_id] for option_id in all_option_ids]

            out_doc = {
                "query": doc["question"],
                "choices": all_options,
                "gold": all_option_ids.index(doc["answer_idx"].strip()),
            }

        else:
            raise ValueError("Unknown dataset path: {}".format(self.DATASET_PATH))

        return out_doc

    # def doc_to_text(self, doc):
    #     "Only question. Ref: OpenbookQA"
    #     return doc["query"]

    def doc_to_text(self, doc):
        return "Question: {}\nAnswer:".format(doc["query"])

    # def doc_to_text(self, doc):
    #     template = ("The following is a multiple-choice question (with options) about medical knowledge. "
    #         "Please select the most appropriate one from the following answer candidates as the final answer.\n\n"
    #         "### Question: {question}\n\n"
    #         "### Options: {options}\n\n"
    #         "### Answer:"
    #     )
    #     return template.format(
    #         question=doc["query"],
    #         options=" ".join(f"({'ABCDE'[i]}) {option}" for i, option in enumerate(doc["choices"])),
    #     )

    # def construct_requests(self, doc, ctx):
    #     lls = [
    #         rf.loglikelihood(ctx, f"{choice}")[0] for choice_id, choice in zip('ABCD', doc["choices"])
    #     ]
    #
    #     return lls

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]
