import re
from typing import Dict, List, Tuple


class StaticAnalyzer:
    """
    Static refactoring repair pass used before entering the agentic loop.
    The operations are conservative and syntax-preserving.
    """

    _IMPORT_RE = re.compile(r"^\s*import\s+[\w\.\*]+\s*;\s*$", re.MULTILINE)

    def apply_static_repairs(
        self,
        original_code: str,
        refactored_code: str,
        failure_log: str,
    ) -> Tuple[str, List[Dict[str, str]]]:
        repaired = refactored_code
        actions: List[Dict[str, str]] = []

        repaired, dep_actions = self._restore_missing_imports(original_code, repaired)
        actions.extend(dep_actions)

        repaired, syntax_actions = self._balance_delimiters(repaired)
        actions.extend(syntax_actions)

        repaired, struct_actions = self._apply_structural_fixes(repaired, failure_log)
        actions.extend(struct_actions)

        return repaired, actions

    def _restore_missing_imports(self, original_code: str, refactored_code: str) -> Tuple[str, List[Dict[str, str]]]:
        original_imports = self._IMPORT_RE.findall(original_code)
        if not original_imports:
            return refactored_code, []

        refactored_imports = set(self._IMPORT_RE.findall(refactored_code))
        missing = [imp for imp in original_imports if imp not in refactored_imports]
        if not missing:
            return refactored_code, []

        patched = self._inject_imports(refactored_code, missing)
        actions = [{
            "operation": "dependency_resolution",
            "details": f"Restored {len(missing)} missing import(s)."
        }]
        return patched, actions

    def _balance_delimiters(self, code: str) -> Tuple[str, List[Dict[str, str]]]:
        actions: List[Dict[str, str]] = []
        updated = code
        pairs = {
            "{": "}",
            "(": ")",
            "[": "]",
        }

        for opener, closer in pairs.items():
            open_count = updated.count(opener)
            close_count = updated.count(closer)
            if open_count > close_count:
                diff = open_count - close_count
                updated = updated.rstrip() + ("\n" + closer * diff) + "\n"
                actions.append({
                    "operation": "syntax_correction",
                    "details": f"Inserted {diff} missing '{closer}' delimiter(s)."
                })

        return updated, actions

    def _apply_structural_fixes(self, code: str, failure_log: str) -> Tuple[str, List[Dict[str, str]]]:
        actions: List[Dict[str, str]] = []
        updated = code
        lower_log = failure_log.lower()

        if "incompatible types" in lower_log or "cannot be converted" in lower_log:
            before = updated
            updated = self._fix_quoted_primitive_assignments(updated)
            if updated != before:
                actions.append({
                    "operation": "structural_correction",
                    "details": "Repaired obvious quoted primitive assignments (int/long/float/double/boolean)."
                })

        return updated, actions

    def _inject_imports(self, code: str, missing_imports: List[str]) -> str:
        package_match = re.search(r"^\s*package\s+[\w\.]+\s*;\s*$", code, re.MULTILINE)
        if package_match:
            insert_pos = package_match.end()
            insertion = "\n\n" + "\n".join(missing_imports)
            return code[:insert_pos] + insertion + code[insert_pos:]

        first_import = re.search(r"^\s*import\s+[\w\.\*]+\s*;\s*$", code, re.MULTILINE)
        if first_import:
            insert_pos = first_import.start()
            insertion = "\n".join(missing_imports) + "\n"
            return code[:insert_pos] + insertion + code[insert_pos:]

        return "\n".join(missing_imports) + "\n\n" + code

    def _fix_quoted_primitive_assignments(self, code: str) -> str:
        rules = [
            (r"(\b(?:int|long|short|byte)\s+\w+\s*=\s*)\"(-?\d+)\"(\s*;)", r"\1\2\3"),
            (r"(\b(?:float|double)\s+\w+\s*=\s*)\"(-?\d+(?:\.\d+)?)\"(\s*;)", r"\1\2\3"),
            (r"(\bboolean\s+\w+\s*=\s*)\"(true|false)\"(\s*;)", r"\1\2\3"),
        ]
        updated = code
        for pattern, replacement in rules:
            updated = re.sub(pattern, replacement, updated, flags=re.IGNORECASE)
        return updated
