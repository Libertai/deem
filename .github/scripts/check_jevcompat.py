"""Gate a jevcompat JSON report: fail on any failed requirement not listed in
--allow, on an aborted run, and on an inconclusive or untested MUST."""

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument(
        "--allow", action="append", default=[], help="requirement id allowed to fail"
    )
    args = parser.parse_args()

    with open(args.report) as f:
        report = json.load(f)

    problems = []
    if report.get("aborted"):
        problems.append(f"run aborted: {report['aborted']}")
    for item in report.get("inconclusive") or []:
        problems.append(f"inconclusive: {item}")
    for req in report["summary"].get("untested_musts") or []:
        problems.append(f"untested MUST: {req}")

    failed = {r["id"] for r in report["results"] if r["status"] == "fail"}
    for req in sorted(failed - set(args.allow)):
        problems.append(f"failed: {req}")
    for req in sorted(set(args.allow) - failed):
        print(f"note: {req} is allowed to fail but passed; drop it from --allow")

    summary = report["summary"]
    print(
        f"jevcompat {report['tool_version']} spec {report['spec_version']}: MUST {summary['MUST']}, SHOULD {summary['SHOULD']}"
    )
    for p in problems:
        print(f"::error::{p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
