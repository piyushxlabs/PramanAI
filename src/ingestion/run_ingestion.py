"""CLI Script to ingest, chunk, embed, and index Uttarakhand Government Order PDFs into PostgreSQL pgvector.

Pure-VLM Strict Ingestion Pipeline:
- Every document is parsed via high-resolution vision preprocessing and sovereign Qwen2.5-VL.
- Atomic per-document processing: if any single page fails VLM extraction, the entire document
  is rejected, all generated chunks are discarded, and zero records are committed to PostgreSQL.
- Fast, memory-bounded batching with immediate pgvector chunk insertion.

Usage:
    python -m src.ingestion.run_ingestion
    python -m src.ingestion.run_ingestion --data-dir path/to/pdfs
    python -m src.ingestion.run_ingestion --file 12201894933.pdf
"""
import sys
import os
# Force UTF-8 output on Windows so box-drawing/Devanagari chars don't crash CP1252
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

import argparse
import traceback
from pathlib import Path

from src.gov_pdf_extractor.pipeline import VlmExtractionError
from src.ingestion.chunking import DocumentChunker
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.vector_store import VectorStore


def run_ingestion_pipeline(
    data_dir: str = "data/raw_pdfs",
    batch_size: int = 4,
    target_file: str | None = None,
) -> dict[str, int]:
    """Processes PDF files in data_dir and indexes them into PostgreSQL pgvector under pure-VLM mode."""
    input_path = Path(data_dir)
    if not input_path.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.", file=sys.stderr)
        return {"processed_docs": 0, "total_pages": 0, "total_chunks": 0}

    pdf_files = sorted(list(input_path.glob("*.pdf")))
    if target_file:
        target_name = Path(target_file).name.lower()
        pdf_files = [
            p for p in pdf_files
            if p.name.lower() == target_name or p.stem.lower() == Path(target_file).stem.lower()
        ]
        if not pdf_files:
            print(f"Error: Specified target file '{target_file}' not found in '{data_dir}'.", file=sys.stderr)
            return {"processed_docs": 0, "total_pages": 0, "total_chunks": 0}

    total_pdf_count = len(pdf_files)

    print("=== ShasanAI Ingestion Pipeline (Pure-VLM Strict Mode: 300+ DPI Floor) ===")
    print(f"Found {total_pdf_count} PDF document(s) to process in '{input_path.resolve()}'")
    print(f"Batch size: {batch_size} | Backend: GovPdfExtractor (Pure-VLM Qwen2.5-VL)")

    parser = PDFParser()
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    store = VectorStore()
    store.initialize_schema()

    if target_file:
        # Wipe only chunks for the targeted document
        print(f"\nWiping old chunks for '{pdf_files[0].name}' from PostgreSQL document_chunks table …")
        with store.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM document_chunks WHERE document_id = %s OR document_id LIKE %s;",
                    (pdf_files[0].name, f"%{pdf_files[0].stem}%"),
                )
        print("Selective document wipe complete.")
    else:
        # Wipe legacy / unnormalised chunks so the DB starts clean
        print("\nWiping old chunks from PostgreSQL document_chunks table …")
        store.clear_all_chunks()
        print("Table truncated.")

    total_pages: int = 0
    total_chunks: int = 0
    processed_docs: int = 0
    failed_docs: list[str] = []

    print("\n" + "-" * 65)

    for idx, pdf_file in enumerate(pdf_files, start=1):
        print(f"[{idx:02d}/{total_pdf_count}] Parsing  '{pdf_file.name}' via Pure-VLM ...", flush=True)
        try:
            # ── 1. Parse PDF → ParsedDocument (Strict VLM extraction per page)
            doc = parser.parse_pdf(pdf_file)

            # ── 2. Chunk → list[DocumentChunk]
            doc_chunks = chunker.chunk_document(doc)

            if not doc_chunks:
                print(
                    f"[{idx:02d}/{total_pdf_count}] WARNING: '{pdf_file.name}' produced 0 chunks "
                    f"(blank/unsupported content) — skipped.",
                    file=sys.stderr,
                )
                failed_docs.append(pdf_file.name)
                continue

            # ── 3. Register parent document record in documents table
            store.insert_document_record(
                document_id=doc.document_id,
                go_number=doc.go_number,
                issuing_department=doc.issuing_department,
                issuing_authority="उत्तराखण्ड शासन",
                date=doc.date or "Unknown",
                total_pages=doc.total_pages,
                status="CURRENT_ACTIVE",
                subject=None,
                file_path=str(doc.filepath),
                ocr_quality_score=1.0,
            )

            # ── 4. Embed + insert chunks immediately (no accumulation in RAM)
            inserted = store.insert_chunks(doc_chunks, batch_size=batch_size)

            total_pages += doc.total_pages
            total_chunks += inserted
            processed_docs += 1

            print(
                f"[{idx:02d}/{total_pdf_count}] SUCCESS  '{pdf_file.name}' -- "
                f"{doc.total_pages} pages | {len(doc_chunks)} chunks | "
                f"Dept: {doc.issuing_department} | GO: {doc.go_number}",
                flush=True,
            )

        except VlmExtractionError as vlm_exc:
            print(
                f"\n[{idx:02d}/{total_pdf_count}] ✗ VLM EXTRACTION REJECTED '{pdf_file.name}': {vlm_exc}",
                file=sys.stderr,
            )
            print("  Atomic Rollback: 0 chunks committed to database for this document.", file=sys.stderr)
            failed_docs.append(pdf_file.name)
            continue
        except Exception as exc:
            print(
                f"\n[{idx:02d}/{total_pdf_count}] ✗ ERROR parsing '{pdf_file.name}': {exc}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            print(file=sys.stderr)
            failed_docs.append(pdf_file.name)
            continue

    print("-" * 65 + "\n")

    # ── Final live count from DB (authoritative)
    total_in_db = store.count_chunks()

    print("=== Ingestion Complete ===")
    print(f"  Documents processed : {processed_docs}/{total_pdf_count}")
    if failed_docs:
        print(f"  Documents failed    : {len(failed_docs)}")
        for name in failed_docs:
            print(f"    FAILED: {name}")
    print(f"  Total pages parsed  : {total_pages}")
    print(f"  Chunks inserted     : {total_chunks}")
    print(f"  Total in pgvector   : {total_in_db}")

    # ── Post-ingestion verification: print 3 sample chunks
    print("\n=== Post-Ingestion Sample Verification (3 chunks) ===")
    store.print_sample_chunks(n=3)

    return {
        "processed_docs": processed_docs,
        "total_pages": total_pages,
        "total_chunks": total_chunks,
    }


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Ingest and index Government Order PDFs via Pure-VLM.")
    arg_parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw_pdfs",
        help="Path to folder containing PDF files (default: data/raw_pdfs)",
    )
    arg_parser.add_argument(
        "--file",
        "--pdf",
        type=str,
        default=None,
        dest="file",
        help="Process only a specific PDF file name (e.g., 8102018173922.pdf)",
    )
    arg_parser.add_argument(
        "--force-reindex",
        "--reindex",
        action="store_true",
        default=False,
        help="Force reindex: truncate existing chunks and re-embed all documents",
    )
    arg_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for embedding generation (default: 4 for hardware safety)",
    )
    args = arg_parser.parse_args()
    run_ingestion_pipeline(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        target_file=args.file,
    )


if __name__ == "__main__":
    main()
