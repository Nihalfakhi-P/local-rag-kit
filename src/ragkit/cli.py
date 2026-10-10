import argparse
import sys
from pathlib import Path

from ragkit.ollama import OllamaError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragkit", description="Private, fully local RAG.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="index a file or directory (.md, .txt, .pdf)")
    p_ingest.add_argument("path", type=Path)

    p_ask = sub.add_parser("ask", help="ask a question against the index")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=None)

    p_eval = sub.add_parser("eval", help="evaluate retrieval (and answers) on a JSONL set")
    p_eval.add_argument("cases", type=Path)
    p_eval.add_argument("-k", type=int, default=5)
    p_eval.add_argument("--retrieval-only", action="store_true", help="skip LLM calls")

    p_serve = sub.add_parser("serve", help="run the HTTP API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        from ragkit.api import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)
        return 0

    from ragkit.pipeline import RAGPipeline

    pipeline = RAGPipeline.from_settings()
    try:
        if args.command == "ingest":
            if not args.path.exists():
                print(f"error: {args.path} does not exist", file=sys.stderr)
                return 2
            counts = pipeline.ingest_path(args.path)
            for source, n in counts.items():
                print(f"  {n:>4} chunks  {source}")
            print(f"Indexed {len(counts)} documents, {pipeline.store.count()} chunks total.")
        elif args.command == "ask":
            answer = pipeline.ask(args.question, args.top_k)
            print(answer.text)
            if answer.sources:
                print("\nSources:")
                for i, hit in enumerate(answer.sources, start=1):
                    mark = "*" if i in answer.citations else " "
                    print(f" {mark}[{i}] {hit.chunk.source} (chunk {hit.chunk.index})")
            if not answer.grounded:
                print(f"\nwarning: {answer.reason}", file=sys.stderr)
        elif args.command == "eval":
            from ragkit.evaluate import (
                evaluate_answers,
                evaluate_retrieval,
                format_report,
                load_cases,
            )

            cases = load_cases(args.cases)
            retrieval = evaluate_retrieval(pipeline, cases, args.k)
            answers = None if args.retrieval_only else evaluate_answers(pipeline, cases)
            print(format_report(retrieval, answers))
    except OllamaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
