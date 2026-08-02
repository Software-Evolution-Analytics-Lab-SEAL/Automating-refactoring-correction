import os
import re
from typing import Dict, List, Tuple

class KnowledgeBase:
    """
    Stores common fix patterns and repository context used during repair.
    The backing store can be swapped for a database or embedding index without
    changing the retrieval interface.
    """
    def __init__(self):
        # Seed patterns that cover the failure modes the pipeline expects.
        self.entries = [
            {
                "id": 1,
                "failure_type": "test_assertion",
                "language": "java",
                "common_fix": "Check if method signature was changed in the refactor. Update test accordingly.",
                "metadata": {"keywords": ["assertion", "refactoring", "signature"]},
            },
            {
                "id": 2,
                "failure_type": "compilation",
                "language": "java",
                "common_fix": "Reintroduce removed import or fix method reference. Possibly fix method visibility.",
                "metadata": {"keywords": ["compilation", "import", "visibility"]},
            },
            # Additional patterns can be added here as new failure cases appear.
        ]
        self.repo_context_cache: Dict[Tuple[str, str], List[Dict]] = {}

    def retrieve_relevant_entries(self, error_message: str, language: str, top_k: int = 3) -> List[Dict]:
        matches = []
        for entry in self.entries:
            if entry["language"] == language:
                # Check if any metadata keyword appears in error_message
                if any(k in error_message.lower() for k in entry["metadata"]["keywords"]):
                    matches.append(entry)

        return matches[:top_k]

    def retrieve_project_context(
        self,
        repo_root: str,
        symbols: List[str],
        max_snippets: int = 6,
        max_chars: int = 9000,
    ) -> List[Dict]:
        """
        Retrieve repository snippets relevant to symbols extracted from failures.
        Results are cached per (repo_root, symbol) to behave as long-term memory.
        """
        contexts: List[Dict] = []
        seen_paths = set()
        total_chars = 0

        for symbol in symbols:
            symbol = symbol.strip()
            if not symbol:
                continue

            cache_key = (repo_root, symbol)
            cached = self.repo_context_cache.get(cache_key)
            if cached is None:
                cached = self._search_symbol_snippets(repo_root, symbol)
                self.repo_context_cache[cache_key] = cached

            for snippet in cached:
                key = (snippet["path"], snippet.get("line", 0), snippet["symbol"])
                if key in seen_paths:
                    continue
                if total_chars + len(snippet["snippet"]) > max_chars:
                    return contexts
                contexts.append(snippet)
                seen_paths.add(key)
                total_chars += len(snippet["snippet"])
                if len(contexts) >= max_snippets:
                    return contexts

        return contexts

    def _search_symbol_snippets(self, repo_root: str, symbol: str) -> List[Dict]:
        symbol_re = re.compile(rf"\b{re.escape(symbol)}\b")
        results: List[Dict] = []

        for root, _, files in os.walk(repo_root):
            for filename in files:
                if not filename.endswith(".java"):
                    continue
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as fh:
                        lines = fh.readlines()
                except (OSError, UnicodeDecodeError):
                    continue

                for idx, line in enumerate(lines):
                    if not symbol_re.search(line):
                        continue
                    start = max(0, idx - 8)
                    end = min(len(lines), idx + 9)
                    snippet = "".join(lines[start:end]).strip()
                    rel_path = os.path.relpath(file_path, repo_root)
                    results.append({
                        "symbol": symbol,
                        "path": rel_path,
                        "line": idx + 1,
                        "snippet": snippet,
                    })
                    break

        results.sort(key=lambda s: (s["path"], s["line"]))
        return results
