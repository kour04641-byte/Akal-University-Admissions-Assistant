from pathlib import Path


def load_documents(data_dir):
    """Read every .txt / .md file in the data folder."""
    docs = []
    data_dir = Path(data_dir)
    for path in sorted(list(data_dir.glob("*.txt")) + list(data_dir.glob("*.md"))):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append({"source": path.name, "text": text})
    return docs