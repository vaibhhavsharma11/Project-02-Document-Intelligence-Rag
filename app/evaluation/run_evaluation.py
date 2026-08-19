from dataclasses import dataclass

from app.evaluation.retrieval_evaluation import (
    evaluate_retrieval,
)


DOCUMENT_ID = (
    "4de34518-297f-45bb-b735-8c1c182559ca"
)


@dataclass
class EvaluationCase:
    name: str
    query: str
    expected_document_id: str
    expected_chunk_index: int | None
    should_retrieve: bool


EVALUATION_CASES = [
    EvaluationCase(
        name="Relevant technology query",
        query=(
            "What technologies can be used to build "
            "a document intelligence and RAG application?"
        ),
        expected_document_id=DOCUMENT_ID,
        expected_chunk_index=1,
        should_retrieve=True,
    ),
    EvaluationCase(
        name="Unsupported salary query",
        query="What is the employee salary structure?",
        expected_document_id=DOCUMENT_ID,
        expected_chunk_index=1,
        should_retrieve=False,
    ),
]


def run_evaluation() -> tuple[int, int]:
    """
    Run all retrieval evaluation cases and return
    the number of passed and failed cases.
    """

    passed_count = 0
    failed_count = 0

    print("=" * 60)
    print("Document Intelligence & RAG Evaluation")
    print("=" * 60)

    for case in EVALUATION_CASES:
        result = evaluate_retrieval(
            query=case.query,
            expected_document_id=(
                case.expected_document_id
            ),
            expected_chunk_index=(
                case.expected_chunk_index
            ),
            top_k=2,
        )

        if case.should_retrieve:
            case_passed = result.passed
        else:
            case_passed = (
                result.retrieved_count == 0
            )

        status = "PASS" if case_passed else "FAIL"

        print()
        print(f"[{status}] {case.name}")
        print(f"Query: {case.query}")
        print(
            f"Retrieved chunks: "
            f"{result.retrieved_count}"
        )
        print(
            f"Top distance: "
            f"{result.top_distance}"
        )
        print(
            f"Top relevance score: "
            f"{result.top_relevance_score}"
        )

        if case_passed:
            passed_count += 1
        else:
            failed_count += 1

    print()
    print("=" * 60)
    print(
        f"Evaluation summary: "
        f"{passed_count} passed, "
        f"{failed_count} failed"
    )
    print("=" * 60)

    return passed_count, failed_count


if __name__ == "__main__":
    passed, failed = run_evaluation()

    if failed > 0:
        raise SystemExit(1)