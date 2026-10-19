# Ultra judge sanity run (FEEDBACK, "What worked well")

Reproduces offline from the committed cassettes in `tests/fixtures/cassettes/judge/` (real Nemotron Ultra responses). The inputs are 20 synthetic classification tickets written for the test (`tests/harness/test_judge_sanity.py`): 12 correct candidates, 6 plainly wrong, 1 right label wrapped in prose, 1 right label with a trailing period.

```bash
uv run python - <<'PY'
import sys
sys.path.insert(0, "tests/harness")
import test_judge_sanity as t
from offramp.tf import TFClient
det, judged, ag = t._agreement(TFClient(cache_dir=t.CASSETTES, live=False))
print(ag)
for d, j in zip(det, judged.scores):
    print(d.case_id, d.passed, round(j.candidate, 2))
PY
```

Output (judge reference score is 5.0 on every case; the deterministic label scorer's pass flag is the second column):

```
Agreement(n=20, agreement=0.95, consistency=1.0)
test_cls:001 .. 012   True   5.0
test_cls:013 .. 018   False  2.33
test_cls:019          False  3.67   (right label wrapped in prose)
test_cls:020          True   4.33   ("Billing.", trailing period)
```

Agreement 19 of 20 and position-swap consistency 100%. The one disagreement is `test_cls:020`: the label scorer forgives the trailing period, while the judge's format criterion scored the candidate 4.33 against the reference's 5.0, so it fails the rule "at least the reference score minus 0.3" (4.33 is under 4.7). The committed test asserts only the targets (agreement at least 0.85, consistency at least 0.90).
