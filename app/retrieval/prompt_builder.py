SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant answering questions using ONLY the provided "
    "context. If the answer isn't in the context, say you don't know. Always "
    "cite the page number(s) your answer came from."
)

def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context_blocks = []
    for match in retrieved_chunks:
        meta = match.get("metadata", {})
        page = meta.get("page", "?")
        text = meta.get("text", "")
        context_blocks.append(f"[Page {page}]\n{text}")

    context_str = "\n".join(context_blocks) if context_blocks else "No relevant context found."

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"--- CONTEXT ---\n{context_str}\n\n"
        f"--- QUESTION ---\n{question}\n"
    )
