"""Checks the port of the official QASPER answer-F1 scoring (offline, no generation)."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("ae", Path(__file__).resolve().parent.parent / "evaluation/qasper/answer_eval.py")
ae = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ae)


def ann(**kw):
    base = {"unanswerable": False, "extractive_spans": [], "free_form_answer": "", "yes_no": None}
    return {**base, **kw}


def test_normalisation_and_token_f1():
    assert ae.normalize_answer("The  BERT-base, model!") == "bertbase model"
    assert ae.token_f1("a BERT model", "the bert model") == 1.0
    assert ae.token_f1("3,044 sentences", "3044 sentences in 100 dialogs") == pytest.approx(2 * (2 / 2) * (2 / 5) / (1 + 2 / 5))
    assert ae.token_f1("", "anything") == 0.0


def test_reference_types():
    assert ae.reference(ann(unanswerable=True)) == ("Unanswerable", "none")
    assert ae.reference(ann(extractive_spans=["x", "y"])) == ("x, y", "extractive")
    assert ae.reference(ann(free_form_answer="free text")) == ("free text", "abstractive")
    assert ae.reference(ann(yes_no=True)) == ("Yes", "boolean")
    assert ae.reference(ann(yes_no=False)) == ("No", "boolean")


def test_max_over_annotations():
    anns = [ann(extractive_spans=["100 dialogs"]), ann(free_form_answer="3,044 sentences")]
    assert ae.answer_f1("3,044 sentences", anns) == 1.0
    assert ae.answer_f1("Unanswerable", [ann(unanswerable=True), ann(yes_no=True)]) == 1.0
