from ingestion.chunking import build_chunks, split_chunk_for_retry, split_into_sections
from ingestion.loaders import LoadedDocument


def test_split_into_sections_detects_article_headings() -> None:
    text = "Article 1 Scope\nThis Regulation applies.\n\nArticle 2 Definitions\nPersonal data means data."
    sections = split_into_sections(text)
    assert len(sections) == 2
    assert sections[0].title == "Article 1 Scope"
    assert sections[1].title == "Article 2 Definitions"


def test_build_chunks_preserves_section_metadata() -> None:
    document = LoadedDocument(
        name="sample.md",
        path="/tmp/sample.md",
        file_type=".md",
        text="Section 1 Intro\n" + ("A" * 1200),
    )
    chunks = build_chunks(document, max_chars=500, overlap_chars=50)
    assert chunks
    assert all(chunk.section_title == "Section 1 Intro" for chunk in chunks)
    assert chunks[0].chunk_id.startswith("sample.md:sec-0000:0:")


def test_split_chunk_for_retry_creates_smaller_children() -> None:
    document = LoadedDocument(
        name="sample.md",
        path="/tmp/sample.md",
        file_type=".md",
        text="Section 1 Intro\n\n" + ("Paragraph one. " * 120) + "\n\n" + ("Paragraph two. " * 120),
    )
    chunk = build_chunks(document, max_chars=4000, overlap_chars=100)[0]
    children = split_chunk_for_retry(chunk, max_chars=1200)
    assert len(children) == 2
    assert all(child.section_title == chunk.section_title for child in children)
    assert all(len(child.text) < len(chunk.text) for child in children)
