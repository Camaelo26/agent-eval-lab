"""Command line entry point.

    python cli.py run --agent v2
    python cli.py run --agent v1 --baseline baselines/v2.json   # should fail
    python cli.py baseline --agent v2                           # record a baseline
    python cli.py audit-judge

Exit code 0 means the gate passed. Anything else means CI should stop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agents import get_agent
from evalkit import gate, judge as judge_mod
from evalkit.dataset import load_jsonl
from evalkit.runner import run_eval

ROOT = Path(__file__).parent
GOLDEN = ROOT / "data" / "golden.jsonl"
TRAPS = ROOT / "data" / "judge_trap_set.jsonl"
DEFAULT_BASELINE = ROOT / "baselines" / "support-agent.json"


def _load_traps() -> list[dict]:
    rows = []
    for line in TRAPS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def _build_report(agent_name: str, judge_name: str):
    agent = get_agent(agent_name)
    cases = load_jsonl(GOLDEN)
    judge = judge_mod.get_judge(judge_name)
    audit = judge_mod.audit_judge(judge, _load_traps())
    return run_eval(agent, cases, judge, suite=f"support-agent:{agent_name}", judge_audit=audit)


def cmd_run(args: argparse.Namespace) -> int:
    report = _build_report(args.agent, args.judge)
    baseline = gate.load_baseline(args.baseline) if args.baseline else None
    result = gate.check(report, gate.Thresholds(), baseline)

    _print_report(report, result)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nreport written to {args.json_out}")
    return result.exit_code


def cmd_baseline(args: argparse.Namespace) -> int:
    report = _build_report(args.agent, args.judge)
    gate.save_baseline(report, args.out)
    print(f"baseline for {report.suite} written to {args.out}")
    print(f"  pass_rate={report.pass_rate:.3f} groundedness={report.mean_groundedness:.3f}")
    return 0


def cmd_audit_judge(args: argparse.Namespace) -> int:
    judge = judge_mod.get_judge(args.judge)
    audit = judge_mod.audit_judge(judge, _load_traps())
    print(f"judge: {judge.name}")
    print(f"  cases          {audit.total}")
    print(f"  accuracy       {audit.accuracy:.3f}")
    print(f"  false passes   {audit.false_pass}  (rate {audit.false_pass_rate:.3f})")
    print(f"  false fails    {audit.false_fail}")
    print(f"  trustworthy    {audit.trustworthy()}")
    return 0 if audit.trustworthy() else 1


def _print_report(report, result) -> None:
    print(f"suite            {report.suite}")
    print(f"judge            {report.judge}")
    print(f"cases            {report.n_cases}")
    print(f"pass rate        {report.pass_rate:.3f}")
    print(f"mean judge score {report.mean_judge_score:.3f}")
    print(f"token f1         {report.mean_token_f1:.3f}")
    print(f"groundedness     {report.mean_groundedness:.3f}")
    print(f"tool score       {report.mean_tool_score:.3f}")
    print(f"latency p50/p95  {report.p50_latency_ms:.2f}ms / {report.p95_latency_ms:.2f}ms")
    print(f"suite cost       ${report.total_cost_usd:.6f}")

    if report.judge_audit:
        audit = report.judge_audit
        print(
            f"judge audit      accuracy {audit['accuracy']:.2f}, "
            f"false pass rate {audit['false_pass_rate']:.2f}, "
            f"trustworthy {audit['trustworthy']}"
        )

    print("\npass rate by tag")
    for tag, rate in report.by_tag.items():
        flag = "  <-- weak" if rate < 0.8 else ""
        print(f"  {tag:<14} {rate:.3f}{flag}")

    if report.failures:
        print(f"\nfailing cases    {', '.join(report.failures)}")

    print("\nGATE " + ("PASS" if result.passed else "FAIL"))
    for violation in result.violations:
        print(f"  - {violation}")


def main(argv: list[str] | None = None) -> int:
    # --judge is shared by every subcommand and accepted on either side of it,
    # so `cli.py run --judge rubric` and `cli.py --judge rubric run` both work.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--judge", default="auto", choices=["auto", "rubric", "anthropic"])

    parser = argparse.ArgumentParser(prog="agent-eval-lab", parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", parents=[common], help="run the suite and apply the gate")
    run.add_argument("--agent", default="v2")
    run.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    run.add_argument("--json-out", default=None)
    run.set_defaults(func=cmd_run)

    baseline = sub.add_parser(
        "baseline", parents=[common], help="record the current run as the baseline"
    )
    baseline.add_argument("--agent", default="v2")
    baseline.add_argument("--out", default=str(DEFAULT_BASELINE))
    baseline.set_defaults(func=cmd_baseline)

    audit = sub.add_parser(
        "audit-judge", parents=[common], help="score the judge against the trap set"
    )
    audit.set_defaults(func=cmd_audit_judge)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
