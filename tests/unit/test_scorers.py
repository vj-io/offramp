"""Deterministic scorers (PRD §6.5, §12.1) and the site-level pass rule (Global Constraint 11).

The scorers decide parity, so these tests lean on over-pass guards: every table has rows where a
sloppy implementation would say "match" and the correct answer is "no".
"""

import json
import random

import pytest

from offramp.evals.scorers import aggregate, json_schema, label, reasoning, score_case, tool_call
from offramp.evals.scorers.aggregate import Thresholds, site_pass
from offramp.schemas import CaseScore, Contract, EvalCase, ModelOutput, RunConfig, ToolCall


def _case(task: str = "x") -> EvalCase:
    return EvalCase(
        id=f"cs_01:{task}",
        site_id="cs_01",
        input_messages=[{"role": "user", "content": "hi"}],
        expected_kind="text",
        split="dev",
    )


def _text(text: str | None) -> ModelOutput:
    return ModelOutput(text=text)


def _calls(*calls: tuple[str, dict]) -> ModelOutput:
    return ModelOutput(tool_calls=[ToolCall(name=n, arguments=a) for n, a in calls])


def _run(scorer, baseline: ModelOutput, candidate: ModelOutput, contract: Contract) -> CaseScore:
    return scorer.score(_case(), baseline, candidate, contract)


# ------------------------------------------------------------------ json_schema

BASE = '{"name": "Alice", "age": 30}'
SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}
ANY_OBJECT = {"type": "object"}
TWENTY = {f"k{i:02d}": i for i in range(20)}


def _twenty(**changed) -> str:
    return json.dumps({**TWENTY, **changed})


# (id, baseline, candidate, contract schema, passed, error_category, score)
JSON_ROWS = [
    ("identical", BASE, BASE, None, True, None, 1.0),
    ("key_order", BASE, '{"age": 30, "name": "Alice"}', None, True, None, 1.0),
    ("string_whitespace_and_case", BASE, '{"name": "  alice ", "age": 30}', None, True, None, 1.0),
    ("int_vs_float_30.0", BASE, '{"name":"Alice","age":30.0}', SCHEMA, True, None, 1.0),
    ("int_vs_exponent", '{"v": 1000}', '{"v": 1e3}', None, True, None, 1.0),
    ("float_trailing_zero", '{"v": 5.1}', '{"v": 5.10}', None, True, None, 1.0),
    ("unicode_casefold", '{"c": "Straße"}', '{"c": "STRASSE"}', None, True, None, 1.0),
    ("scalar_root", '"yes"', '" YES "', None, True, None, 1.0),
    ("fence_json", BASE, f"```json\n{BASE}\n```", None, True, None, 1.0),
    ("fence_no_tag", BASE, f"```\n{BASE}\n```", None, True, None, 1.0),
    ("fence_upper_tag_and_padding", BASE, f"\n  ```JSON\n{BASE}\n```  \n", None, True, None, 1.0),
    ("fence_on_baseline_too", f"```json\n{BASE}\n```", BASE, None, True, None, 1.0),
    ("nested_equal", '{"u":{"n":"A","t":["x","y"]}}', '{"u":{"t":["X","Y"],"n":"a"}}', None, True, None, 1.0),
    ("one_wrong_of_twenty_is_boundary_pass", _twenty(), _twenty(k07=99), ANY_OBJECT, True, None, 0.95),
    # --- must fail
    ("prose_before_fence", BASE, f"Here you go:\n```json\n{BASE}\n```", None, False, "invalid_json", 0.0),
    ("trailing_prose", BASE, BASE + " hope this helps", None, False, "invalid_json", 0.0),
    ("not_json", BASE, "Alice, 30", None, False, "invalid_json", 0.0),
    ("nan_is_not_json", BASE, '{"name":"Alice","age":NaN}', None, False, "invalid_json", 0.0),
    ("empty", BASE, "", None, False, "empty_output", 0.0),
    ("whitespace_only", BASE, "  \n ", None, False, "empty_output", 0.0),
    ("none_text", BASE, None, None, False, "empty_output", 0.0),
    ("string_for_int_explicit", BASE, '{"name":"Alice","age":"30"}', SCHEMA, False, "schema_violation", 0),
    ("string_for_number_inferred", BASE, '{"name":"Alice","age":"30"}', None, False, "schema_violation", 0.0),
    ("bool_for_number_inferred", BASE, '{"name":"Alice","age":true}', None, False, "schema_violation", 0.0),
    ("missing_required_explicit", BASE, '{"name":"Alice"}', SCHEMA, False, "schema_violation", 0.0),
    ("missing_required_inferred", BASE, '{"name":"Alice"}', None, False, "schema_violation", 0.0),
    ("broken_contract_schema", BASE, BASE, {"type": "nonsense"}, False, "schema_error", 0.0),
    ("dangling_ref_in_schema", BASE, BASE, {"$ref": "#/$defs/nope"}, False, "schema_error", 0.0),
    ("baseline_unusable", "not json at all", BASE, None, False, "baseline_unusable", 0.0),
    ("missing_key", '{"a":1,"b":2}', '{"a":1}', ANY_OBJECT, False, "missing_key", 0.5),
    ("extra_key_is_penalised", '{"a":1,"b":2,"c":3}', '{"a":1,"b":2,"c":3,"d":4}', ANY_OBJECT, False,
     "extra_key", 0.75),
    ("wrong_value", '{"a":1,"b":2}', '{"a":1,"b":3}', ANY_OBJECT, False, "wrong_value", 0.5),
    ("two_wrong_of_twenty", _twenty(), _twenty(k01=99, k02=99), ANY_OBJECT, False, "wrong_value", 0.9),
    ("bool_is_not_one", '{"f": true}', '{"f": 1}', ANY_OBJECT, False, "wrong_value", 0.0),
    ("one_is_not_bool", '{"f": 1}', '{"f": true}', ANY_OBJECT, False, "wrong_value", 0.0),
    ("numeric_string_is_not_number", '{"n": 5}', '{"n": "5"}', ANY_OBJECT, False, "wrong_value", 0.0),
    ("overflow_exponent_never_equal", '{"v": 1e400}', '{"v": 2e400}', ANY_OBJECT, False, "wrong_value", 0.0),
    ("big_int_precision", '{"id": 12345678901234567890}', '{"id": 12345678901234567891}', ANY_OBJECT,
     False, "wrong_value", 0.0),
    ("null_is_not_missing", '{"a":1,"b":null}', '{"a":1}', ANY_OBJECT, False, "missing_key", 0.5),
    ("empty_list_is_a_key", '{"a":1,"items":[]}', '{"a":1}', ANY_OBJECT, False, "missing_key", 0.5),
    ("empty_object_is_a_key", '{"a":1,"meta":{}}', '{"a":1}', ANY_OBJECT, False, "missing_key", 0.5),
    ("missing_and_wrong_reports_missing", '{"a":1,"b":2,"c":3}', '{"a":1,"b":9}', ANY_OBJECT, False,
     "missing_key", 1 / 3),
    ("list_order_matters", '{"t":["x","y"]}', '{"t":["y","x"]}', ANY_OBJECT, False, "wrong_value", 0.0),
    ("list_gains_item", '{"t":[1,2]}', '{"t":[1,2,3]}', ANY_OBJECT, False, "extra_key", 2 / 3),
    ("object_replaced_by_scalar", '{"a":{"b":1}}', '{"a":5}', ANY_OBJECT, False, "missing_key", 0.0),
    ("root_array_reordered", "[1,2]", "[2,1]", None, False, "wrong_value", 0.0),
]


@pytest.mark.parametrize(
    "baseline,candidate,schema,passed,category,score",
    [r[1:] for r in JSON_ROWS],
    ids=[r[0] for r in JSON_ROWS],
)
def test_json_schema_scorer(baseline, candidate, schema, passed, category, score):
    contract = Contract(task_type="json_extraction", output="json_schema", json_schema=schema)
    got = _run(json_schema, _text(baseline), _text(candidate), contract)
    assert got.case_id == "cs_01:x"
    assert (got.passed, got.error_category) == (passed, category)
    assert got.score == pytest.approx(score)


def test_json_schema_deeply_nested_candidate_fails_instead_of_crashing():
    contract = Contract(task_type="json_extraction", output="json")
    got = _run(json_schema, _text(BASE), _text("[" * 100_000 + "]" * 100_000), contract)
    assert (got.passed, got.error_category) == (False, "invalid_json")


def test_json_schema_failure_detail_names_the_paths():
    contract = Contract(task_type="json_extraction", output="json", json_schema=ANY_OBJECT)
    got = _run(json_schema, _text('{"a":1,"b":{"c":2}}'), _text('{"a":9,"x":1}'), contract)
    assert "missing: b.c" in got.detail and "wrong: a" in got.detail and "extra: x" in got.detail


# ------------------------------------------------------------------ label

LABELS = ["Billing", "Tech Support", "spam", "not spam"]

# (id, baseline, candidate, labels, passed, error_category)
LABEL_ROWS = [
    ("identical", "positive", "positive", [], True, None),
    ("case", "Positive", "POSITIVE", [], True, None),
    ("whitespace", "positive", "  positive\n", [], True, None),
    ("trailing_period", "positive", "Positive.", [], True, None),
    ("several_trailing_marks", "positive", "positive!!!", [], True, None),
    ("surrounding_quotes", "positive", '"positive"', [], True, None),
    ("markdown_bold", "positive", "**positive**", [], True, None),
    ("casefold_not_lower", "Straße", "STRASSE", [], True, None),
    ("mapped_to_label_set", "Tech Support", "tech support.", LABELS, True, None),
    ("baseline_also_normalised", "billing.", "BILLING", LABELS, True, None),
    ("punctuation_only_label_kept", "?", "?", ["?", "!"], True, None),
    ("same_offlabel_value_is_agreement", "neutral", "Neutral", LABELS, True, None),
    # --- must fail
    ("different", "positive", "negative", [], False, "wrong_label"),
    ("different_in_label_set", "Billing", "spam", LABELS, False, "wrong_label"),
    ("offlabel_candidate", "Billing", "refund request", LABELS, False, "invalid_label"),
    ("no_substring_matching", "spam", "not spam", LABELS, False, "wrong_label"),
    ("negation_prefix", "positive", "not positive", [], False, "wrong_label"),
    ("two_labels", "positive", "positive, negative", [], False, "wrong_label"),
    ("prose_around_label", "positive", "The label is positive", [], False, "wrong_label"),
    ("punctuation_only_labels_differ", "?", "!", ["?", "!"], False, "wrong_label"),
    ("both_empty_is_not_agreement", "", "", [], False, "empty_output"),
    ("empty_candidate", "positive", "", [], False, "empty_output"),
    ("none_candidate_text", "positive", None, [], False, "empty_output"),
    ("whitespace_only_candidate", "positive", " \n", [], False, "empty_output"),
    ("only_punctuation_candidate_vs_word", "positive", "...", [], False, "wrong_label"),
    ("empty_baseline", "", "positive", [], False, "baseline_unusable"),
    ("none_baseline", None, "positive", [], False, "baseline_unusable"),
]


@pytest.mark.parametrize(
    "baseline,candidate,labels,passed,category",
    [r[1:] for r in LABEL_ROWS],
    ids=[r[0] for r in LABEL_ROWS],
)
def test_label_scorer(baseline, candidate, labels, passed, category):
    contract = Contract(task_type="classification", output="label", labels=labels)
    got = _run(label, _text(baseline), _text(candidate), contract)
    assert (got.passed, got.error_category) == (passed, category)
    assert got.score == (1.0 if passed else 0.0)


# ------------------------------------------------------------------ tool_call

W = ("get_weather", {"city": "Paris", "units": "metric"})
SET = ("set_temp", {"value": 5})

# (id, baseline calls, candidate calls, passed, error_category)
TOOL_ROWS = [
    ("identical", [W], [W], True, None),
    ("arg_key_order", [W], [("get_weather", {"units": "metric", "city": "Paris"})], True, None),
    ("arg_whitespace_and_case", [W], [("get_weather", {"city": "  paris ", "units": "METRIC"})], True, None),
    ("number_5_vs_str", [SET], [("set_temp", {"value": "5"})], True, None),
    ("number_5_vs_float", [SET], [("set_temp", {"value": 5.0})], True, None),
    ("number_5_vs_str_float", [SET], [("set_temp", {"value": "5.0"})], True, None),
    ("number_5_vs_padded_str", [SET], [("set_temp", {"value": " 5.00 "})], True, None),
    ("number_str_baseline_vs_number", [("set_temp", {"value": "5"})], [SET], True, None),
    ("number_str_float_baseline_vs_number", [("set_temp", {"value": "5.0"})], [SET], True, None),
    ("number_float_vs_str_float", [("f", {"v": 5.1})], [("f", {"v": "5.10"})], True, None),
    ("negative_str", [("f", {"v": -5})], [("f", {"v": "-5"})], True, None),
    ("nested_args", [("q", {"f": {"tags": ["A", "b"], "n": 1}})],
     [("q", {"f": {"n": 1.0, "tags": ["a", " B"]}})], True, None),
    ("two_calls_in_order", [W, SET], [W, SET], True, None),
    # --- must fail
    ("wrong_tool", [W], [("get_forecast", dict(W[1]))], False, "wrong_tool"),
    ("tool_name_case", [W], [("Get_Weather", dict(W[1]))], False, "wrong_tool"),
    ("wrong_arg_value", [W], [("get_weather", {"city": "Rome", "units": "metric"})], False, "wrong_args"),
    ("extra_arg", [W], [("get_weather", {**W[1], "lang": "fr"})], False, "wrong_args"),
    ("missing_arg", [W], [("get_weather", {"city": "Paris"})], False, "wrong_args"),
    ("null_is_not_missing", [("f", {"a": 1})], [("f", {"a": 1, "b": None})], False, "wrong_args"),
    ("number_differs", [SET], [("set_temp", {"value": 6})], False, "wrong_args"),
    ("leading_zero_string_is_not_number", [SET], [("set_temp", {"value": "05"})], False, "wrong_args"),
    ("word_is_not_number", [SET], [("set_temp", {"value": "five"})], False, "wrong_args"),
    ("bool_is_not_number", [SET], [("set_temp", {"value": True})], False, "wrong_args"),
    ("bool_is_not_one", [("f", {"v": 1})], [("f", {"v": True})], False, "wrong_args"),
    ("number_is_not_bool", [("f", {"v": True})], [("f", {"v": 1})], False, "wrong_args"),
    ("bool_is_not_string", [("f", {"v": True})], [("f", {"v": "true"})], False, "wrong_args"),
    ("zip_leading_zero_lost", [("f", {"zip": "01234"})], [("f", {"zip": 1234})], False, "wrong_args"),
    ("zip_leading_zero_gained", [("f", {"zip": 1234})], [("f", {"zip": "01234"})], False, "wrong_args"),
    ("zip_leading_zero_kept", [("f", {"zip": "01234"})], [("f", {"zip": "01234"})], True, None),
    ("big_int_precision", [("f", {"id": 12345678901234567890})], [("f", {"id": "12345678901234567891"})],
     False, "wrong_args"),
    ("exponent_string_stays_string", [("f", {"v": 1000})], [("f", {"v": "1e3"})], False, "wrong_args"),
    ("nan_never_equal", [("f", {"v": float("nan")})], [("f", {"v": float("nan")})], False, "wrong_args"),
    ("inf_never_equal", [("f", {"v": float("inf")})], [("f", {"v": float("inf")})], False, "wrong_args"),
    ("non_ascii_digit_is_not_number", [("f", {"v": 55})], [("f", {"v": "5\u0665"})], False, "wrong_args"),
    ("list_order_matters", [("f", {"v": [1, 2]})], [("f", {"v": [2, 1]})], False, "wrong_args"),
    ("no_call_when_expected", [W], [], False, "no_tool_call"),
    ("call_when_none_expected", [], [W], False, "wrong_tool"),
    ("fewer_calls", [W, SET], [W], False, "wrong_tool"),
    ("more_calls", [W], [W, SET], False, "wrong_tool"),
    ("calls_swapped", [W, SET], [SET, W], False, "wrong_tool"),
    ("second_call_args_differ", [W, SET], [W, ("set_temp", {"value": 6})], False, "wrong_args"),
]


@pytest.mark.parametrize(
    "baseline,candidate,passed,category",
    [r[1:] for r in TOOL_ROWS],
    ids=[r[0] for r in TOOL_ROWS],
)
def test_tool_call_scorer(baseline, candidate, passed, category):
    contract = Contract(task_type="tool_call", output="tool_call", tools=True)
    got = _run(tool_call, _calls(*baseline), _calls(*candidate), contract)
    assert (got.passed, got.error_category) == (passed, category)
    assert got.score == (1.0 if passed else 0.0)


def test_tool_call_ignores_text_channel():
    contract = Contract(task_type="tool_call", output="tool_call", tools=True)
    base = ModelOutput(text="calling it", tool_calls=[ToolCall(name=W[0], arguments=W[1])])
    cand = ModelOutput(text=None, tool_calls=[ToolCall(name=W[0], arguments=W[1])])
    assert _run(tool_call, base, cand, contract).passed


@pytest.mark.parametrize("baseline_text", [None, "I can answer directly: 42"], ids=["ref_blank", "ref_text"])
@pytest.mark.parametrize(
    "candidate_text,passed,category",
    [
        ("42, no tool needed", True, None),
        (None, False, "empty_output"),
        ("", False, "empty_output"),
        (" \n\t", False, "empty_output"),
    ],
    ids=["answers", "none", "empty", "whitespace"],
)
def test_tool_call_abstention_needs_a_non_blank_candidate(baseline_text, candidate_text, passed, category):
    # Neither side calls a tool: a truncated or blank candidate is not a correct abstention.
    contract = Contract(task_type="tool_call", output="tool_call", tools=True)
    got = _run(tool_call, _text(baseline_text), _text(candidate_text), contract)
    assert (got.passed, got.error_category) == (passed, category)
    assert got.score == (1.0 if passed else 0.0)


# ------------------------------------------------------------------ reasoning

# (id, baseline text, candidate text, passed, error_category)
REASONING_ROWS = [
    ("answer_line", "so 6*7\nANSWER: 42", "ANSWER: 42", True, None),
    ("answer_lowercase_and_period", "ANSWER: 42", "answer: 42.", True, None),
    ("answer_bold_label", "ANSWER: 42", "**ANSWER:** 42", True, None),
    ("answer_bold_value", "ANSWER: 42", "ANSWER: **42**", True, None),
    ("answer_crlf", "ANSWER: 42", "thinking\r\nANSWER: 42\r\n", True, None),
    ("last_answer_line_wins", "ANSWER: 42", "ANSWER: 7\nhmm, redo\nANSWER: 42", True, None),
    ("answer_line_beats_later_number", "ANSWER: 7", "ANSWER: 7\nbut then I counted 42 apples", True, None),
    ("fallback_last_number", "ANSWER: 42", "Step 1: 3+4 = 7. Step 2: 7*6 = 42", True, None),
    ("fallback_ignores_trailing_words", "ANSWER: 42", "so 6*7 = 42 apples", True, None),
    ("fallback_trailing_period", "ANSWER: 42", "The result is 42.", True, None),
    ("fallback_on_baseline", "the total is 1,234", "ANSWER: 1234", True, None),
    ("fallback_leading_dot_number", "ANSWER: 0.5", "so it equals .5", True, None),
    ("fallback_negative_after_equals", "ANSWER: -4", "so x=-4", True, None),
    ("fallback_hyphen_is_not_a_sign", "ANSWER: 4", "between 3-4", True, None),
    ("thousands_separator", "ANSWER: 1,234", "ANSWER: 1234", True, None),
    ("currency_and_decimal", "ANSWER: $1,234.50", "ANSWER: 1234.5", True, None),
    ("unicode_minus", "ANSWER: -5", "ANSWER: \u22125", True, None),
    ("leading_dot", "ANSWER: 0.5", "ANSWER: .5", True, None),
    ("exponent", "ANSWER: 1e3", "ANSWER: 1000", True, None),
    ("tol_inside", "ANSWER: 1", "ANSWER: 1.0000001", True, None),
    ("tol_inside_pi", "ANSWER: 3.14159265", "ANSWER: 3.1415926", True, None),
    ("zero_equals_zero", "ANSWER: 0", "ANSWER: 0.0", True, None),
    ("exact_text", "ANSWER: B", "ANSWER: B", True, None),
    ("exact_text_trailing_period", "ANSWER: Paris.", "ANSWER: Paris", True, None),
    # --- must fail
    ("different_number", "ANSWER: 42", "ANSWER: 43", False, "wrong_answer"),
    ("first_answer_line_loses", "ANSWER: 42", "ANSWER: 42\nwait, no\nANSWER: 7", False, "wrong_answer"),
    ("answer_line_beats_later_number_2", "ANSWER: 42", "ANSWER: 7\nbut final: 42", False, "wrong_answer"),
    ("sign_matters", "ANSWER: -5", "ANSWER: 5", False, "wrong_answer"),
    ("tol_outside", "ANSWER: 1", "ANSWER: 1.00001", False, "wrong_answer"),
    ("zero_has_no_absolute_tolerance", "ANSWER: 0", "ANSWER: 0.000000001", False, "wrong_answer"),
    ("european_decimal_is_not_a_number", "ANSWER: 1,5", "ANSWER: 1.5", False, "wrong_answer"),
    ("fraction_is_not_decimal", "ANSWER: 1/2", "ANSWER: 0.5", False, "wrong_answer"),
    ("units_break_exactness", "ANSWER: 42 apples", "ANSWER: 42", False, "wrong_answer"),
    ("overflow_is_not_equal", "ANSWER: 1e999", "ANSWER: 2e999", False, "wrong_answer"),
    ("non_ascii_digit_is_not_number", "ANSWER: 5", "ANSWER: \u0665", False, "wrong_answer"),
    ("exact_is_case_sensitive", "ANSWER: B", "ANSWER: b", False, "wrong_answer"),
    ("exact_keeps_parentheses", "ANSWER: B", "ANSWER: (B)", False, "wrong_answer"),
    ("exact_text_differs", "ANSWER: yes", "ANSWER: no", False, "wrong_answer"),
    ("number_vs_text", "ANSWER: 42", "ANSWER: forty-two", False, "wrong_answer"),
    ("no_answer_no_number", "ANSWER: 42", "I cannot determine this.", False, "no_answer"),
    ("digits_glued_to_letters_are_not_a_number", "ANSWER: 2", "the molecule is H2O", False, "no_answer"),
    ("empty_candidate", "ANSWER: 42", "", False, "no_answer"),
    ("none_candidate", "ANSWER: 42", None, False, "no_answer"),
    ("both_unextractable_is_not_agreement", "No idea.", "No idea.", False, "no_answer"),
    ("baseline_unextractable", "No idea.", "ANSWER: 42", False, "baseline_unusable"),
    ("baseline_none", None, "ANSWER: 42", False, "baseline_unusable"),
]


@pytest.mark.parametrize(
    "baseline,candidate,passed,category",
    [r[1:] for r in REASONING_ROWS],
    ids=[r[0] for r in REASONING_ROWS],
)
def test_reasoning_scorer(baseline, candidate, passed, category):
    contract = Contract(task_type="reasoning", output="text", reasoning_heavy=True)
    got = _run(reasoning, _text(baseline), _text(candidate), contract)
    assert (got.passed, got.error_category) == (passed, category)
    assert got.score == (1.0 if passed else 0.0)


# ------------------------------------------------------------------ score_case dispatch


@pytest.mark.parametrize(
    "contract,baseline,candidate",
    [
        (Contract(task_type="classification", output="label"), _text("a"), _text("a")),
        (Contract(task_type="json_extraction", output="json"), _text(BASE), _text(BASE)),
        (Contract(task_type="tool_call", output="tool_call", tools=True), _calls(W), _calls(W)),
        (Contract(task_type="reasoning", output="text"), _text("ANSWER: 1"), _text("ANSWER: 1")),
    ],
    ids=["classification", "json_extraction", "tool_call", "reasoning"],
)
def test_score_case_routes_by_task_type(contract, baseline, candidate):
    assert score_case(_case(), baseline, candidate, contract).passed


def test_score_case_refuses_free_text():
    contract = Contract(task_type="free_text", output="text")
    with pytest.raises(ValueError, match="judge"):
        score_case(_case(), _text("a"), _text("a"), contract)


@pytest.mark.parametrize("task", ["classification", "json_extraction", "tool_call", "reasoning"])
def test_score_case_scores_a_failed_candidate_call_as_a_failure(task):
    contract = Contract(task_type=task, output="text")
    got = score_case(_case(), _text("a"), None, contract)
    assert (got.passed, got.score, got.error_category) == (False, 0.0, aggregate.CANDIDATE_ERROR)


# ------------------------------------------------------------------ aggregate.site_pass


def _scores(n_pass: int, fail_categories: list[str]) -> list[CaseScore]:
    out = [CaseScore(case_id=f"p{i}", passed=True, score=1.0) for i in range(n_pass)]
    out += [
        CaseScore(case_id=f"f{i}", passed=False, score=0.0, error_category=c)
        for i, c in enumerate(fail_categories)
    ]
    return out


def _judge(*values: float) -> list[CaseScore]:
    return [CaseScore(case_id=f"j{i}", passed=v >= 4.0, score=v) for i, v in enumerate(values)]


# (id, task_type, n_pass, fail categories, passed, score)
FRACTION_ROWS = [
    ("cls_exactly_95", "classification", 19, ["wrong_label"], True, 0.95),
    ("cls_90", "classification", 18, ["wrong_label"] * 2, False, 0.90),
    ("cls_perfect", "classification", 20, [], True, 1.0),
    ("cls_none_pass", "classification", 0, ["wrong_label"] * 3, False, 0.0),
    ("tool_exactly_95", "tool_call", 19, ["wrong_args"], True, 0.95),
    ("tool_90", "tool_call", 18, ["wrong_tool"] * 2, False, 0.90),
    ("reasoning_exactly_90", "reasoning", 18, ["wrong_answer"] * 2, True, 0.90),
    ("reasoning_85", "reasoning", 17, ["wrong_answer"] * 3, False, 0.85),
    ("json_95_all_valid", "json_extraction", 19, ["field_mismatch"], True, 0.95),
    ("json_90_all_valid", "json_extraction", 18, ["field_mismatch"] * 2, False, 0.90),
    ("json_one_invalid_json", "json_extraction", 19, ["invalid_json"], False, 0.95),
    ("json_one_schema_violation", "json_extraction", 19, ["schema_violation"], False, 0.95),
    ("json_one_broken_contract", "json_extraction", 19, ["schema_error"], False, 0.95),
    ("json_one_empty_output", "json_extraction", 19, ["empty_output"], False, 0.95),
    ("json_one_failed_call", "json_extraction", 19, [aggregate.CANDIDATE_ERROR], False, 0.95),
    ("json_unusable_baseline_is_a_failed_case", "json_extraction", 19, ["baseline_unusable"], True, 0.95),
    ("json_perfect", "json_extraction", 5, [], True, 1.0),
]


@pytest.mark.parametrize(
    "task,n_pass,fails,passed,score",
    [r[1:] for r in FRACTION_ROWS],
    ids=[r[0] for r in FRACTION_ROWS],
)
def test_site_pass_default_thresholds(task, n_pass, fails, passed, score):
    got_passed, got_score = site_pass(task, _scores(n_pass, fails))
    assert got_passed is passed
    assert got_score == pytest.approx(score)


@pytest.mark.parametrize("task", ["classification", "json_extraction", "tool_call", "reasoning", "free_text"])
def test_site_pass_with_no_cases_never_passes(task):
    assert site_pass(task, [], Thresholds(0.0, 0.0), baseline_judge_mean=1.0) == (False, 0.0)


def test_thresholds_are_parameters_and_split_by_task_family():
    eighteen = _scores(18, ["wrong"] * 2)  # 0.90
    assert site_pass("classification", eighteen)[0] is False
    assert site_pass("classification", eighteen, Thresholds(threshold=0.90))[0] is True
    assert site_pass("classification", eighteen, Thresholds(reasoning_threshold=0.5))[0] is False
    assert site_pass("reasoning", eighteen)[0] is True
    assert site_pass("reasoning", eighteen, Thresholds(reasoning_threshold=0.95))[0] is False
    assert site_pass("reasoning", eighteen, Thresholds(threshold=0.99))[0] is True
    assert Thresholds() == Thresholds(threshold=0.95, reasoning_threshold=0.90)


def _borderline(n: int, score: float = 0.95) -> list[CaseScore]:
    """n json cases that each passed with the given field match."""
    return [CaseScore(case_id=f"b{i}", passed=True, score=score) for i in range(n)]


def test_json_site_rule_gates_the_mean_field_match():
    # 19 cases at exactly the 0.95 per-case bar plus one with every value wrong: 95% of cases pass,
    # but the mean field match is 0.9025, below the 95% the rule asks for.
    scores = [*_borderline(19), CaseScore(case_id="f", passed=False, score=0.0, error_category="wrong_value")]
    passed, score = site_pass("json_extraction", scores)
    assert passed is False
    assert score == pytest.approx(0.9025)


def test_json_site_threshold_applies_to_the_mean_field_match():
    scores = _borderline(20)  # every case has one wrong field in twenty
    passed, score = site_pass("json_extraction", scores)
    assert passed is True
    assert score == pytest.approx(0.95)  # the reported score is the field-match rate, not "100% of cases"
    stricter = [Thresholds(threshold=1.0), Thresholds(threshold=0.96), RunConfig(repo_url="x", threshold=1.0)]
    for thresholds in stricter:
        assert site_pass("json_extraction", scores, thresholds)[0] is False
    assert site_pass("json_extraction", _borderline(20, 1.0), Thresholds(threshold=1.0)) == (True, 1.0)


def test_json_site_rule_still_needs_the_fraction_of_passing_cases():
    # mean field match 0.97, but half the cases are below the per-case bar
    scores = [
        *_borderline(10, 1.0),
        *[CaseScore(case_id=f"f{i}", passed=False, score=0.94, error_category="x") for i in range(10)],
    ]
    passed, score = site_pass("json_extraction", scores)
    assert passed is False
    assert score == pytest.approx(0.97)


def test_site_pass_counts_passed_flags_not_scores():
    scores = [CaseScore(case_id=str(i), passed=False, score=0.99) for i in range(10)]
    assert site_pass("classification", scores) == (False, 0.0)


def test_site_pass_accepts_a_run_config():
    cfg = RunConfig(repo_url="https://github.com/x/y", threshold=0.90, reasoning_threshold=0.95)
    eighteen = _scores(18, ["wrong"] * 2)
    assert site_pass("classification", eighteen, cfg)[0] is True
    assert site_pass("reasoning", eighteen, cfg)[0] is False


# (id, judge scores, baseline mean, passed)
JUDGE_ROWS = [
    ("equal_to_baseline_above_floor", [4.5] * 10, 4.5, True),
    ("exactly_baseline_minus_margin", [4.2] * 10, 4.5, True),
    ("just_under_margin", [4.19] * 10, 4.5, False),
    ("mean_of_mixed_scores_at_margin", [4, 5] * 5, 4.8, True),
    ("floor_applies_even_when_baseline_is_low", [3.9] * 5, 3.9, False),
    ("exactly_floor_and_baseline", [4.0] * 5, 4.0, True),
    ("float_noise_boundary_4.4", [4.1] * 10, 4.4, True),  # 4.4 - 0.3 is 4.1000000000000005 in floats
    ("float_noise_boundary_4.9", [4.6] * 10, 4.9, True),
    ("floor_ok_but_baseline_gap_too_big", [4.0] * 5, 4.5, False),
    ("better_than_baseline", [5.0] * 5, 4.1, True),
    ("perfect_vs_perfect", [5.0] * 5, 5.0, True),
    ("mean_not_median", [5, 5, 5, 5, 1], 4.6, False),
]


@pytest.mark.parametrize(
    "judge,baseline_mean,passed",
    [r[1:] for r in JUDGE_ROWS],
    ids=[r[0] for r in JUDGE_ROWS],
)
def test_site_pass_free_text(judge, baseline_mean, passed):
    got_passed, got_score = site_pass("free_text", _judge(*judge), Thresholds(), baseline_mean)
    assert got_passed is passed
    assert got_score == pytest.approx(sum(judge) / len(judge))


def test_free_text_ignores_fraction_thresholds():
    got = site_pass("free_text", _judge(4.2, 4.2), Thresholds(threshold=1.0, reasoning_threshold=1.0), 4.5)
    assert got[0] is True


@pytest.mark.parametrize(
    "judge,passed",
    [([4.0] * 3, True), ([4.5, 3.5], True), ([5.0] * 3, True), ([3.9] * 3, False), ([5, 5, 5, 1, 1], False)],
    ids=["at_floor", "mean_at_floor", "perfect", "just_under_floor", "mean_under_floor"],
)
def test_free_text_without_a_reference_mean_applies_only_the_floor(judge, passed):
    explicit_none = site_pass("free_text", _judge(*judge), Thresholds(), None)
    for got in (site_pass("free_text", _judge(*judge)), explicit_none):
        assert got[0] is passed
        assert got[1] == pytest.approx(sum(judge) / len(judge))


def test_a_reference_mean_adds_the_margin_on_top_of_the_floor():
    assert site_pass("free_text", _judge(4.0, 4.0))[0] is True
    assert site_pass("free_text", _judge(4.0, 4.0), baseline_judge_mean=4.3)[0] is True
    assert site_pass("free_text", _judge(4.0, 4.0), baseline_judge_mean=4.4)[0] is False


@pytest.mark.parametrize("bad", [50.0, 5.01, 0.99, 0.0, -1.0, float("nan"), float("inf")])
def test_free_text_rejects_judge_scores_outside_the_rubric(bad):
    with pytest.raises(ValueError, match="judge score"):
        site_pass("free_text", _judge(5.0, 5.0, bad))
    with pytest.raises(ValueError, match="judge score"):
        site_pass("free_text", _judge(5.0, 5.0, bad), Thresholds(), 4.0)


def test_free_text_accepts_the_rubric_endpoints():
    assert site_pass("free_text", _judge(1.0, 5.0)) == (False, 3.0)
    assert site_pass("free_text", _judge(5.0, 5.0)) == (True, 5.0)


@pytest.mark.parametrize("bad", [0.0, 0.99, 5.01, float("nan")])
def test_free_text_rejects_a_reference_mean_outside_the_rubric(bad):
    with pytest.raises(ValueError, match="baseline_judge_mean"):
        site_pass("free_text", _judge(5.0), Thresholds(), bad)


# ------------------------------------------------------------------ properties

GRID = [i / 20 for i in range(21)]  # 0.00, 0.05 ... 1.00
_FAILURES = ["wrong_label", "invalid_json", "schema_violation", "field_mismatch", aggregate.CANDIDATE_ERROR]


def _random_scores(rng: random.Random) -> list[CaseScore]:
    out = []
    for i in range(rng.randint(1, 40)):
        ok = rng.random() < rng.choice([0.5, 0.9, 0.97, 1.0])
        category = None if ok else rng.choice(_FAILURES)
        score = rng.choice([0.95, 0.97, 1.0]) if ok else rng.choice([0.0, 0.5, 0.9])  # json field matches
        out.append(CaseScore(case_id=f"c{i}", passed=ok, score=score, error_category=category))
    return out


@pytest.mark.parametrize("task", ["classification", "json_extraction", "tool_call", "reasoning"])
@pytest.mark.parametrize("knob", ["threshold", "reasoning_threshold"])
def test_raising_thresholds_never_flips_fail_to_pass(task, knob):
    rng = random.Random(20260519)
    for _ in range(300):
        scores = _random_scores(rng)
        passes = [site_pass(task, scores, Thresholds(**{knob: t}))[0] for t in GRID]
        assert passes == sorted(passes, reverse=True), (task, knob, scores)  # True...True, False...False


def test_raising_the_judge_baseline_never_flips_fail_to_pass():
    rng = random.Random(7)
    for _ in range(300):
        scores = _judge(*(rng.choice([1, 2, 3, 4, 5]) for _ in range(rng.randint(1, 12))))
        passes = [site_pass("free_text", scores, Thresholds(), b / 10)[0] for b in range(10, 51)]
        assert passes == sorted(passes, reverse=True)  # True...True, False...False


# ------------------------------------------------------------------ end to end: garbage never passes


@pytest.mark.parametrize(
    "contract,baseline,garbage",
    [
        (Contract(task_type="classification", output="label", labels=["a", "b"]), _text("a"), _text("")),
        (Contract(task_type="json_extraction", output="json"), _text(BASE), _text("lorem ipsum")),
        (Contract(task_type="tool_call", output="tool_call", tools=True), _calls(W), _calls()),
        (Contract(task_type="tool_call", output="tool_call", tools=True), _text("I can answer"), _text(None)),
        (Contract(task_type="reasoning", output="text"), _text("ANSWER: 42"), _text("I do not know")),
    ],
    ids=["classification", "json_extraction", "tool_call", "tool_call_abstain", "reasoning"],
)
def test_garbage_candidate_cannot_pass_a_site(contract, baseline, garbage):
    scores = [score_case(_case(str(i)), baseline, garbage, contract) for i in range(20)]
    assert site_pass(contract.task_type, scores)[0] is False


def test_matching_candidate_passes_a_site_end_to_end():
    contract = Contract(task_type="classification", output="label", labels=["a", "b"])
    scores = [score_case(_case(str(i)), _text("a"), _text("A."), contract) for i in range(20)]
    scores[3] = score_case(_case("3"), _text("a"), _text("b"), contract)
    assert site_pass("classification", scores) == (True, 0.95)


def test_json_site_rule_end_to_end_uses_the_field_match_rate():
    contract = Contract(task_type="json_extraction", output="json", json_schema=ANY_OBJECT)
    cases = [_case(str(i)) for i in range(20)]
    one_wrong = [score_case(c, _text(_twenty()), _text(_twenty(k07=99)), contract) for c in cases]
    assert all(c.passed and c.score == pytest.approx(0.95) for c in one_wrong)
    assert site_pass("json_extraction", one_wrong)[0] is True
    assert site_pass("json_extraction", one_wrong, Thresholds(threshold=1.0))[0] is False

    all_wrong = score_case(_case("x"), _text(_twenty()), _text(json.dumps({k: -1 for k in TWENTY})), contract)
    mixed = [*one_wrong[:19], all_wrong]
    assert site_pass("json_extraction", mixed)[0] is False
