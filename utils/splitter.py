import re


def split_documents(docs, chunk_size=1200, overlap=150):
    """Split documents into chunks.

    Every '## ' heading starts a new section. A section that fits in chunk_size
    stays whole (so a fee table is never cut in half). Longer sections are cut
    into overlapping pieces, and each piece gets the heading again so it keeps
    its context.
    """
    chunks = []
    for doc in docs:
        sections = re.split(r"\n(?=## )", doc["text"])
        for section in sections:
            section = section.strip()
            if not section:
                continue
            heading = section.splitlines()[0]
            if len(section) <= chunk_size:
                chunks.append({"source": doc["source"], "text": section})
                continue
            start = 0
            while start < len(section):
                piece = section[start:start + chunk_size]
                if start > 0:
                    piece = heading + "\n" + piece
                chunks.append({"source": doc["source"], "text": piece})
                if start + chunk_size >= len(section):
                    break
                start += chunk_size - overlap
    return chunks