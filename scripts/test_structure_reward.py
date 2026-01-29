#!/usr/bin/env python3
"""
Quick sanity tests for `structure_reward` in src/train/reward_funcs.py.

Run:
  python scripts/test_structure_reward.py
"""

from __future__ import annotations

from dataclasses import dataclass
import sys


# Ensure `src/` is on sys.path so we can import `train.reward_funcs` when running from repo root.
try:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
except Exception:
    pass


from train.reward_funcs import structure_reward  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    text: str
    expect_reward: float


def _wrap_completion(text: str):
    # reward funcs expect: completions = [ [ {"content": "..."} ], ... ]
    return [[{"content": text}]]


def main() -> int:
    cases = [
        Case(
            name="valid_minimal",
            text="<think>a</think><answer>1</answer>",
            expect_reward=1.0,
        ),
        Case(
            name="valid_with_whitespace",
            text="  \n<think>reasoning</think>\n\n<answer>123</answer>\n",
            expect_reward=1.0,
        ),
        Case(
            name="invalid_first_token_not_think",
            text="hello <think>x</think><answer>1</answer>",
            expect_reward=-1.0,
        ),
        Case(
            name="invalid_missing_answer",
            text="<think>x</think>",
            expect_reward=-1.0,
        ),
        Case(
            name="invalid_missing_think",
            text="<answer>1</answer>",
            expect_reward=-1.0,
        ),
        Case(
            name="invalid_trailing_text_after_answer",
            text="<think>x</think><answer>1</answer> extra",
            expect_reward=-1.0,
        ),
        Case(
            name="invalid_stray_think_close_inside_answer",
            text="<think>I think the answer is</think><answer>1 because i am right </think></answer>",
            expect_reward=-1.0,
        ),
        Case(
            name="invalid_nested_answer_inside_think",
            text="<think>foo <answer>bar</answer></think><answer>1</answer>",
            expect_reward=-1.0,
        ),
        Case(
            name="invalid_extra_tags_after_answer",
            text="<think>x</think><answer>1</answer><answer>2</answer>",
            expect_reward=-1.0,
        ),
        Case(
            name="invalid_extra_open_think_inside_think",
            text="<think>oops <think>nested</think></think><answer>1</answer>",
            expect_reward=-1.0,
        ),
        Case(
            name="invalid_characters between think and answer",
            text="<think>x</think>t<answer>1</answer>",
            expect_reward=-1.0,
        )
    ]

    failures = 0
    for c in cases:
        completions = _wrap_completion(c.text)
        # `assistant` is unused by structure_reward; pass a dummy list of same length.
        assistant = [{"content": ""}]

        got = structure_reward(completions=completions, assistant=assistant)[0]
        ok = got == c.expect_reward
        status = "PASS" if ok else "FAIL"
        print(f"{status}  {c.name}: got={got} expected={c.expect_reward}")
        if not ok:
            failures += 1

    if failures:
        print(f"\n{failures} failing case(s).")
        return 1

    print("\nAll cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

