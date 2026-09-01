"""Database Audit CLI for Uttarakhand Government Order Chunks.

Inspects PostgreSQL document_chunks table, printing column metadata,
document distribution, and clean decoded Devanagari text excerpts.

Usage:
    python -m src.tools.audit_chunks
    python -m src.tools.audit_chunks --limit 10
    python -m src.tools.audit_chunks --go-number "146/XXVII(1)/2018"
    python -m src.tools.audit_chunks --doc-id "24122020162747.pdf"
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Optional

from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row

# Force UTF-8 output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

load_dotenv()


def get_db_url() -> str:
    """Gets and normalizes PostgreSQL connection URL from environment."""
    raw_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/shasanai")
    return re.sub(r"\+.*?://", "://", raw_url)


def audit_database(
    limit: int = 25,
    go_number: Optional[str] = None,
    doc_id: Optional[str] = None,
    dept: Optional[str] = None,
    page: Optional[int] = None,
) -> None:
    """Queries and displays PostgreSQL document_chunks metadata and Devanagari content."""
    db_url = get_db_url()

    try:
        conn = psycopg.connect(db_url, row_factory=dict_row)
    except Exception as exc:
        print(f"ERROR: Could not connect to PostgreSQL database at {db_url}: {exc}", file=sys.stderr)
        return

    with conn:
        with conn.cursor() as cur:
            # 1. Total chunk count
            cur.execute("SELECT COUNT(*) AS total FROM document_chunks;")
            row = cur.fetchone()
            total_chunks = row["total"] if row else 0

            print("\n" + "=" * 80)
            print(f"=== SHASANAI POSTGRESQL CHUNKS AUDIT (Total: {total_chunks} Chunks) ===")
            print("=" * 80)

            if total_chunks == 0:
                print("No records found in document_chunks table. Run ingestion first:\n  python -m src.ingestion.run_ingestion")
                return

            # 2. Table Column Schema
            cur.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'document_chunks'
                ORDER BY ordinal_position;
                """
            )
            col_rows = cur.fetchall()
            col_names = [r["column_name"] for r in col_rows]
            print(f"TABLE COLUMNS ({len(col_names)}): {', '.join(col_names)}\n")

            # 3. Document / GO Summary
            cur.execute(
                """
                SELECT
                    document_id,
                    go_number,
                    issuing_department,
                    date,
                    COUNT(*) AS chunk_count,
                    COUNT(DISTINCT page_number) AS page_count
                FROM document_chunks
                GROUP BY document_id, go_number, issuing_department, date
                ORDER BY document_id ASC;
                """
            )
            doc_summary = cur.fetchall()
            print("--- INDEXED DOCUMENTS SUMMARY ---")
            print(f"{'Document ID':<25} | {'GO Number':<35} | {'Department':<15} | {'Pages':<5} | {'Chunks':<6}")
            print("-" * 95)
            for d in doc_summary:
                print(
                    f"{d['document_id']:<25} | {d['go_number']:<35} | {d['issuing_department']:<15} | "
                    f"{d['page_count']:<5} | {d['chunk_count']:<6}"
                )
            print("-" * 95 + "\n")

            # 4. Detailed Chunk Query
            query = """
                SELECT
                    id,
                    chunk_id,
                    document_id,
                    file_path,
                    go_number,
                    issuing_department,
                    date,
                    page_number,
                    chunk_index,
                    exact_text_excerpt,
                    bounding_box_coordinates,
                    table_json,
                    math_verification_status,
                    font_encoding_type,
                    created_at
                FROM document_chunks
                WHERE 1=1
            """
            params: list[Any] = []

            if go_number:
                query += " AND go_number ILIKE %s"
                params.append(f"%{go_number}%")
            if doc_id:
                query += " AND document_id ILIKE %s"
                params.append(f"%{doc_id}%")
            if dept:
                query += " AND issuing_department ILIKE %s"
                params.append(f"%{dept}%")
            if page is not None:
                query += " AND page_number = %s"
                params.append(page)

            query += " ORDER BY document_id ASC, page_number ASC, chunk_index ASC LIMIT %s;"
            params.append(limit)

            cur.execute(query, params)
            chunks = cur.fetchall()

            print(f"--- DETAILED CHUNKS (Displaying {len(chunks)} of {total_chunks}) ---")
            for c in chunks:
                print(
                    f"\n[ID: {c['id']} | CHUNK: {c['chunk_id']} | PAGE: {c['page_number']} | "
                    f"GO: {c['go_number']} | DEPT: {c['issuing_department']}]"
                )
                if c.get("file_path"):
                    print(f"  FILE: {c['file_path']}")
                if c.get("bounding_box_coordinates"):
                    print(f"  BBOX: {c['bounding_box_coordinates']}")
                if c.get("font_encoding_type"):
                    print(f"  FONT TYPE: {c['font_encoding_type']}")
                if c.get("math_verification_status"):
                    print(f"  MATH STATUS: {c['math_verification_status']}")

                content = c.get("exact_text_excerpt", "")
                print("  CONTENT:")
                print("  " + "\n  ".join(content.splitlines()[:15]))
                if len(content.splitlines()) > 15:
                    print("  ... (truncated for CLI display)")
                print("-" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ShasanAI PostgreSQL document_chunks table.")
    parser.add_argument("--limit", "-n", type=int, default=25, help="Number of chunks to display (default: 25)")
    parser.add_argument("--go-number", "-g", type=str, default=None, help="Filter by GO number substring")
    parser.add_argument("--doc-id", "-d", type=str, default=None, help="Filter by document ID")
    parser.add_argument("--dept", type=str, default=None, help="Filter by issuing department")
    parser.add_argument("--page", "-p", type=int, default=None, help="Filter by page number")
    args = parser.parse_args()

    audit_database(
        limit=args.limit,
        go_number=args.go_number,
        doc_id=args.doc_id,
        dept=args.dept,
        page=args.page,
    )


if __name__ == "__main__":
    main()
