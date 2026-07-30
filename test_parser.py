"""Tests for parser.py — abdominal CT / BTCV scope.

Most tests inject synthetic LLM output via the `_llm_output` kwarg so they
run fast and deterministically. The `LiveOllamaTests` class hits the local
llama3.1:8b model and is skipped unless `--live` is passed.
"""
import json
import logging
import sys
import unittest

from parser import (
    BTCV_ORGANS,
    TaskParams,
    confidence_router,
    expand_abbreviations,
    parse_instruction,
    _extract_json,
)


# ---------------------------------------------------------------------------
# Pure-function tests (no LLM required)
# ---------------------------------------------------------------------------

class ExpandAbbreviationsTests(unittest.TestCase):
    def test_expands_abdominal_anatomy(self):
        self.assertEqual(expand_abbreviations("RK"), "right kidney")
        self.assertEqual(expand_abbreviations("LK"), "left kidney")
        self.assertEqual(expand_abbreviations("IVC"), "inferior vena cava")
        self.assertEqual(expand_abbreviations("GB"), "gallbladder")
        self.assertEqual(expand_abbreviations("panc"), "pancreas")

    def test_expands_pathology_shorthand(self):
        self.assertEqual(expand_abbreviations("mets"), "metastasis")
        self.assertEqual(expand_abbreviations("CA"), "carcinoma")
        self.assertEqual(expand_abbreviations("HCC"), "hepatocellular carcinoma")

    def test_expansions_in_sentence(self):
        self.assertEqual(
            expand_abbreviations("check rk for cysts"),
            "check right kidney for cysts",
        )

    def test_unknown_words_pass_through(self):
        self.assertEqual(
            expand_abbreviations("find a lesion in the liver"),
            "find a lesion in the liver",
        )

    def test_no_substring_collisions(self):
        # "ca" mustn't accidentally rewrite "scan" or "carcinoma"; whole-word
        # matching is the contract — guard it.
        self.assertEqual(expand_abbreviations("scan the cancer"), "scan the cancer")


class BtcvVocabularyTests(unittest.TestCase):
    def test_btcv_has_thirteen_organs(self):
        self.assertEqual(len(BTCV_ORGANS), 13)

    def test_btcv_includes_expected_organs(self):
        for organ in [
            "spleen", "right kidney", "left kidney", "gallbladder",
            "esophagus", "liver", "stomach", "aorta", "inferior vena cava",
            "portal vein", "pancreas", "right adrenal gland",
            "left adrenal gland",
        ]:
            self.assertIn(organ, BTCV_ORGANS)


class ExtractJsonTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(_extract_json('{"organ": "liver"}'), {"organ": "liver"})

    def test_strips_markdown_fences(self):
        wrapped = '```json\n{"organ": "liver", "pathology": "lesion"}\n```'
        self.assertEqual(_extract_json(wrapped)["organ"], "liver")

    def test_strips_preamble(self):
        noisy = 'Here is your JSON:\n{"organ": "spleen", "pathology": "mass"}'
        self.assertEqual(_extract_json(noisy)["organ"], "spleen")

    def test_raises_on_no_json(self):
        with self.assertRaises(ValueError):
            _extract_json("I cannot help with that.")


# ---------------------------------------------------------------------------
# parse_instruction with injected LLM output
# ---------------------------------------------------------------------------

def _json(**kw) -> str:
    return json.dumps(kw)


class ParseInstructionMockedTests(unittest.TestCase):
    def test_liver_lesion(self):
        fake = _json(
            organ="liver", pathology="lesion", task="detect+segment",
            region=None, urgency="routine", parse_confidence="high",
        )
        result = parse_instruction("find a lesion in the liver", _llm_output=fake)
        self.assertEqual(result.organ, "liver")
        self.assertEqual(result.pathology, "lesion")
        self.assertEqual(result.task, "detect+segment")
        self.assertEqual(result.parse_confidence, "high")

    def test_kidney_cyst_with_side(self):
        fake = _json(
            organ="right kidney", pathology="cyst", task="detect+segment",
            region=None, urgency="routine", parse_confidence="high",
        )
        result = parse_instruction("check rk for cysts", _llm_output=fake)
        self.assertEqual(result.organ, "right kidney")
        self.assertEqual(result.pathology, "cyst")

    def test_pancreas_mass_urgent(self):
        fake = _json(
            organ="pancreas", pathology="mass", task="detect+segment",
            region="head", urgency="urgent", parse_confidence="high",
        )
        result = parse_instruction(
            "urgently identify mass in pancreas head", _llm_output=fake,
        )
        self.assertEqual(result.organ, "pancreas")
        self.assertEqual(result.urgency, "urgent")
        self.assertEqual(result.region, "head")

    def test_aorta_calcification(self):
        fake = _json(
            organ="aorta", pathology="calcification", task="detect+segment",
            region=None, urgency="routine", parse_confidence="high",
        )
        result = parse_instruction(
            "look for calcifications along the aorta", _llm_output=fake,
        )
        self.assertEqual(result.organ, "aorta")
        self.assertEqual(result.pathology, "calcification")

    def test_markdown_wrapped_output_is_recovered(self):
        fake = '```json\n' + _json(
            organ="spleen", pathology="lesion", task="segment",
            region=None, urgency="routine", parse_confidence="high",
        ) + '\n```'
        result = parse_instruction("segment the spleen", _llm_output=fake)
        self.assertEqual(result.task, "segment")
        self.assertEqual(result.organ, "spleen")

    def test_malformed_json_returns_low_confidence(self):
        result = parse_instruction("find tumor in liver", _llm_output="not json at all")
        self.assertEqual(result.parse_confidence, "low")
        self.assertEqual(result.organ, "unknown")
        self.assertEqual(result.pathology, "unknown")

    def test_truncated_json_returns_low_confidence(self):
        result = parse_instruction(
            "find lesion in liver",
            _llm_output='{"organ": "liver", "pathology":',
        )
        self.assertEqual(result.parse_confidence, "low")

    def test_schema_violation_returns_low_confidence(self):
        bad = _json(organ="liver", pathology="lesion")  # missing "task"
        result = parse_instruction("find lesion in liver", _llm_output=bad)
        self.assertEqual(result.parse_confidence, "low")


class BtcvOrganValidatorTests(unittest.TestCase):
    """The TaskParams validator should auto-downgrade non-BTCV organs."""

    def test_btcv_organ_keeps_high_confidence(self):
        result = parse_instruction(
            "find lesion in liver",
            _llm_output=_json(
                organ="liver", pathology="lesion", task="detect+segment",
                parse_confidence="high",
            ),
        )
        self.assertEqual(result.parse_confidence, "high")

    def test_brain_organ_downgraded_to_low(self):
        # The LLM might still emit "brain" if the radiologist asks for it;
        # the validator must catch this and route it to rejection.
        result = parse_instruction(
            "find tumor in brain",
            _llm_output=_json(
                organ="brain", pathology="tumor", task="detect+segment",
                parse_confidence="high",
            ),
        )
        self.assertEqual(result.parse_confidence, "low")

    def test_lung_organ_downgraded(self):
        result = parse_instruction(
            "find nodule in lung",
            _llm_output=_json(
                organ="lung", pathology="nodule", task="detect+segment",
                parse_confidence="high",
            ),
        )
        self.assertEqual(result.parse_confidence, "low")

    def test_kidney_without_side_downgraded(self):
        # BTCV requires "left kidney" or "right kidney" — bare "kidney" is
        # ambiguous and should be downgraded.
        result = parse_instruction(
            "find cyst in kidney",
            _llm_output=_json(
                organ="kidney", pathology="cyst", task="detect+segment",
                parse_confidence="high",
            ),
        )
        self.assertEqual(result.parse_confidence, "low")


# ---------------------------------------------------------------------------
# confidence_router
# ---------------------------------------------------------------------------

class ConfidenceRouterTests(unittest.TestCase):
    def _params(self, **overrides) -> TaskParams:
        defaults = dict(
            organ="liver", pathology="lesion", task="detect+segment",
            region=None, urgency="routine", parse_confidence="high",
        )
        defaults.update(overrides)
        return TaskParams(**defaults)

    def test_proceed_on_btcv_organ(self):
        self.assertEqual(confidence_router(self._params()), "proceed")

    def test_proceed_on_each_btcv_organ(self):
        for organ in BTCV_ORGANS:
            self.assertEqual(
                confidence_router(self._params(organ=organ)),
                "proceed",
                msg=f"BTCV organ {organ!r} should proceed",
            )

    def test_reject_on_non_btcv_organ(self):
        # Build TaskParams directly so validator+router both see "brain".
        # The validator will downgrade confidence to "low"; router must
        # still hard-reject.
        p = TaskParams(
            organ="brain", pathology="tumor", task="detect+segment",
            parse_confidence="high",
        )
        self.assertEqual(confidence_router(p), "reject")

    def test_reject_on_unknown_organ(self):
        self.assertEqual(
            confidence_router(self._params(organ="unknown")),
            "reject",
        )

    def test_reject_on_unknown_pathology(self):
        self.assertEqual(
            confidence_router(self._params(pathology="unknown")),
            "reject",
        )

    def test_flag_on_low_confidence_with_btcv_organ(self):
        self.assertEqual(
            confidence_router(self._params(parse_confidence="low")),
            "flag_for_review",
        )


# ---------------------------------------------------------------------------
# End-to-end: instruction → rejected, for non-BTCV organ
# ---------------------------------------------------------------------------

class NonBtcvRejectionTests(unittest.TestCase):
    def test_brain_instruction_is_rejected(self):
        # Even if the LLM cooperates and returns organ="brain", we reject.
        result = parse_instruction(
            "find tumor in brain",
            _llm_output=_json(
                organ="brain", pathology="tumor", task="detect+segment",
                parse_confidence="high",
            ),
        )
        self.assertEqual(result.parse_confidence, "low")
        self.assertEqual(confidence_router(result), "reject")


# ---------------------------------------------------------------------------
# Live integration tests
# ---------------------------------------------------------------------------

RUN_LIVE = False


class LiveOllamaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not RUN_LIVE:
            raise unittest.SkipTest("pass --live to run integration tests")

    def test_liver_lesion(self):
        result = parse_instruction("find a lesion in the liver")
        self.assertEqual(result.organ, "liver")
        self.assertIn("lesion", result.pathology.lower())
        self.assertEqual(confidence_router(result), "proceed")

    def test_right_kidney_cyst_via_abbreviation(self):
        result = parse_instruction("check rk for cysts")
        self.assertEqual(result.organ, "right kidney")

    def test_brain_input_rejected(self):
        result = parse_instruction("find tumor in brain")
        # Either the LLM returns organ="unknown" (per system prompt) or
        # something non-BTCV (validator downgrades). Either way: reject.
        self.assertEqual(confidence_router(result), "reject")


def main():
    global RUN_LIVE
    if "--live" in sys.argv:
        RUN_LIVE = True
        sys.argv.remove("--live")
    logging.basicConfig(level=logging.WARNING)
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
