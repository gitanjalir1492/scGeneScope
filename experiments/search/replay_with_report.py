"""Run the unchanged replay entry point, then report local Chat results offline.

All arguments, environment variables and standard streams pass through to the
execution child. Execution validation and fingerprinting belong to replay_search;
this launcher must not alter inference arguments or implement execution policy.
Reporting imports occur only after the child has finished successfully.
"""

from pathlib import Path
import subprocess
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    execution = Path(__file__).with_name("replay_search.py")
    try:
        subprocess.run([sys.executable, str(execution), *argv], check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
    if "--help" in argv or "-h" in argv:
        return

    import replay_search

    args = replay_search.build_parser().parse_args(argv)
    if args.llm_api == "chat" and not args.analyze_only:
        import llm_analysis

        args.methods = list(dict.fromkeys(args.methods))
        llm_analysis.after_replay(args, replay_search)


if __name__ == "__main__":
    main()
