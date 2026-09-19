"""Single entrypoint: ``python -m rlr <command> [args]``.

Every command module is imported lazily so that ``rlr.env`` (imported by the
package ``__init__``) always runs before transformers / sentence-transformers.
"""

import argparse
import importlib
import sys

# command -> (module, help)
COMMANDS: dict[str, tuple[str, str]] = {
    "env-check": ("rlr.envcheck", "check GPU, cache dirs and library versions"),
    "download": ("rlr.data.download", "download codes from legalacts.ru"),
    "parse": ("rlr.data.parse", "parse raw pages into articles.jsonl"),
    "chunk": ("rlr.data.chunk", "build chunk corpus and corpus stats"),
    "splits": ("rlr.data.splits", "held-out articles, dev/test article selection"),
    "generate": ("rlr.gen.generate", "generate synthetic questions with local LLM"),
    "judge": ("rlr.gen.judge", "LLM judge for dev/test questions"),
    "build-dataset": ("rlr.data.build", "filters, dedup, final train/dev/test files"),
    "export-mteb": ("rlr.data.export_mteb", "export dataset in MTEB retrieval format"),
    "baselines": ("rlr.eval.run_baselines", "evaluate baseline models"),
    "mine": ("rlr.train.mine_negatives", "mine hard negatives"),
    "train": ("rlr.train.train", "fine-tune an embedding model"),
    "evaluate": ("rlr.eval.run_baselines", "evaluate a (fine-tuned) model"),
    "bootstrap": ("rlr.eval.bootstrap", "paired bootstrap significance"),
    "speed": ("rlr.eval.speed", "speed / size benchmark"),
    "forgetting": ("rlr.eval.forgetting", "RuBQ forgetting check"),
    "annotate": ("rlr.annotate.app", "golden set annotation UI"),
    "golden": ("rlr.annotate.golden", "golden subset: select items / build qrels"),
}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="python -m rlr")
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args(argv[:1])
    module = importlib.import_module(COMMANDS[ns.command][0])
    module.main(argv[1:])


if __name__ == "__main__":
    main()
