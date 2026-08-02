import os
import re
import difflib
from typing import List, Dict, Optional, Tuple

from gpt4_wrapper import GPT4Wrapper
from rag import KnowledgeBase
from prompts_manager import PromptsManager
from test_runner import JavaTestRunner
from starcoder_refactoring import perform_starcoder_refactor
from static_analyzer import StaticAnalyzer

class GPT4RefactoringOrchestrator:
    """
    0) Clones & tests the *single test* from the CSV to ensure original code passes.
    1) Calls StarCoder2 for refactoring on 'method_path'.
    2) Retests the single test in an iterative GPT-4 fix loop if it fails.
    """

    def __init__(
        self,
        gpt_api_key: str,
        repo_name: str,
        repo_url: str,
        max_iterations: int = 10,
        clone_dir: str = "./cloned_repos",
        timeout_seconds: int = 600
    ):
        self.gpt4 = GPT4Wrapper(api_key=gpt_api_key)
        self.knowledge_base = KnowledgeBase()
        self.prompts_manager = PromptsManager()

        # "test_runner.py" handles cloning/building/running single tests
        self.test_runner = JavaTestRunner(
            repo_name=repo_name,
            repo_url=repo_url,
            clone_dir=clone_dir,
            timeout_seconds=timeout_seconds
        )
        self.max_iterations = max_iterations
        self.iteration_logs = []
        self.static_analyzer = StaticAnalyzer()
        self.short_term_memory: List[Dict] = []

    def run_refactoring_pipeline(
        self,
        method_path: str,
        method_name: str,
        test_path: str,
        initial_refactor_mode: str = "generate",
        pre_refactored_dir: str = "./refactored_methods",
        language: str = "java"
    ) -> Dict:
        """
        Steps:
        0) clone & build aggregator
        0b) run single test from test_path => must pass or we skip
        1) starcoder2 refactor on method_path
        2) iterative single-test & GPT-4 fix loop
        Returns a dict with:
        success, iterations, final_code, logs (iteration logs)
        """
        result = {
            "success": False,
            "iterations": 0,
            "final_code": "",
            "logs": []
        }
        self.iteration_logs = []
        self.short_term_memory = []

        refactored_methods_dir = pre_refactored_dir
        os.makedirs(refactored_methods_dir, exist_ok=True)

        try:
            # Step 0: Clone + Build
            self.test_runner.clone_or_update_repo()
            top_pom = self.test_runner.find_top_level_pom()
            if top_pom:
                built_ok = self.test_runner.build_entire_project(top_pom)
                if not built_ok:
                    result["logs"].append({"warning": "Aggregator build might fail tests."})
            else:
                result["logs"].append({"error": "No top-level pom.xml found. Attempting submodule approach only."})

            # Step 0b: single test => parse test class & find submodule pom
            test_rel = self.strip_tmp_prefix(test_path)
            test_class = self.derive_test_class_name(test_rel)
            sub_pom = self.find_pom_for_test_file(test_rel)
            pom_to_use = sub_pom
            # fallback to aggregator
            if not sub_pom:
                # fallback to top-level if it exists
                if top_pom and os.path.isfile(top_pom):
                    pom_to_use = top_pom
                    print(f"[INFO] Falling back to top-level pom: {pom_to_use}")
                else:
                    result["logs"].append({"error": "No submodule pom or top-level pom found. Skipping test."})
                    return result

            # Run single test on original code
            success_orig, logs_orig = self.run_single_test(pom_to_use, test_class)
            iteration_0 = {"iteration": 0, "actions": []}
            iteration_0["actions"].append({
                "step": "test_execution",
                "pass": success_orig,
                "output": logs_orig
            })
            result["logs"].append(iteration_0)
            
            # Log the original test result for traceability.
            self.prompts_manager.log_test_step(
                test_class,  # file identifier
                "Original Test Result",
                f"Test Class: {test_class}\nPass: {success_orig}\nLogs:\n{logs_orig}"
            )

            if not success_orig:
                result["logs"].append({"error": "Original code test fails. Not refactoring."})
                return result

            # Step 1: StarCoder2 refactor
            method_rel = self.strip_tmp_prefix(method_path)
            absolute_method = os.path.join(
                self.test_runner.local_repo_path,
                method_rel
            )

            original_code = ""
            if os.path.isfile(absolute_method):
                with open(absolute_method, "r") as f:
                    original_code = f.read()

            # Write the original code to a file
            original_file = os.path.join(refactored_methods_dir, f"{method_name}_original.java")
            with open(original_file, "w") as iter_file:
                iter_file.write(original_code)

            # Step 1: initial StarCoder2 refactor strategy (generate or load precomputed iter1)
            if initial_refactor_mode == "precomputed":
                sc2_iter1 = os.path.join(refactored_methods_dir, f"{method_name}_iter1.java")
                if not (os.path.isfile(sc2_iter1) and os.path.getsize(sc2_iter1) > 0):
                    result["logs"].append({"error": f"No pre-refactored code at {sc2_iter1}; skipping."})
                    return result
                with open(sc2_iter1, "r") as src:
                    refactored_code = src.read()
            elif initial_refactor_mode == "generate":
                refactored_code = perform_starcoder_refactor(
                    method_path=absolute_method,
                    method_name=method_name,
                    file_identifier=method_name
                )  # No fix prompt in first iteration

                if refactored_code.strip() == "cuda":
                    iteration_info = {"iteration": 1, "actions": [{
                        "step": "test_execution",
                        "pass": "cuda",
                        "output": "CUDA out of memory encountered during StarCoder2 generation."
                    }]}
                    result["logs"].append(iteration_info)
                    return result
            else:
                raise ValueError(
                    f"Unsupported initial_refactor_mode '{initial_refactor_mode}'. "
                    "Use 'generate' or 'precomputed'."
                )

            # Write the refactored code back to the repository
            with open(absolute_method, "w") as f:
                f.write(refactored_code)

            # Step 2: evaluate initial refactor once
            current_code = refactored_code
            success, logs_ = self.run_single_test(pom_to_use, test_class)
            initial_iteration = {
                "iteration": 1,
                "actions": [{
                    "step": "test_execution",
                    "pass": success,
                    "output": logs_
                }]
            }
            result["logs"].append(initial_iteration)
            self.prompts_manager.log_test_step(
                test_class,
                "Initial Refactor Test Result",
                f"Test Class: {test_class}\nPass: {success}\nLogs:\n{logs_}"
            )

            # Step 2a: static repair stage before agentic loop
            if not success:
                static_fixed_code, static_actions = self.static_analyzer.apply_static_repairs(
                    original_code=original_code,
                    refactored_code=current_code,
                    failure_log=logs_,
                )
                if static_actions and static_fixed_code != current_code:
                    with open(absolute_method, "w") as wf:
                        wf.write(static_fixed_code)
                    current_code = static_fixed_code
                    static_success, static_logs = self.run_single_test(pom_to_use, test_class)
                    initial_iteration["actions"].append({
                        "step": "static_repair",
                        "operations": static_actions,
                    })
                    initial_iteration["actions"].append({
                        "step": "test_execution_after_static",
                        "pass": static_success,
                        "output": static_logs,
                    })
                    self.prompts_manager.log_test_step(
                        test_class,
                        "Static Repair Test Result",
                        f"Test Class: {test_class}\nPass: {static_success}\nLogs:\n{static_logs}"
                    )
                    success = static_success
                    logs_ = static_logs

            if success:
                iteration_file = os.path.join(refactored_methods_dir, f"{method_name}_iter1.java")
                with open(iteration_file, "w") as iter_file:
                    iter_file.write(current_code)
                result["success"] = True
                result["iterations"] = 1
                result["final_code"] = current_code
                return result

            # Step 3: iterative agentic repair loop
            failure_logs = logs_
            for iteration in range(2, self.max_iterations + 1):
                normalized_log = self._normalize_error_log(failure_logs)
                failure_signals = self._extract_failure_signals(normalized_log)
                symbols = self._collect_symbols(failure_signals, original_code, current_code)
                kb_entries = self.knowledge_base.retrieve_relevant_entries(normalized_log, language)
                project_context_entries = self.knowledge_base.retrieve_project_context(
                    repo_root=self.test_runner.local_repo_path,
                    symbols=symbols,
                )
                project_context = self._format_project_context(project_context_entries)
                diff_str = self._compute_diff(original_code, current_code)
                focused_diff = self._extract_focused_diff(diff_str, symbols)
                signature_text = failure_signals.get("signature", "")
                location_text = "; ".join(failure_signals.get("locations", []))

                explanation_prompt = self.prompts_manager.build_explanation_prompt(
                    refactored_diff=focused_diff,
                    error_log=normalized_log,
                    test_name=f"{method_name}_Test",
                    knowledge_base_entries=kb_entries,
                    error_signature=signature_text,
                    failure_location=location_text,
                    project_context=project_context,
                )
                explanation = self.gpt4.call_gpt4(explanation_prompt)

                starcoder_fix_prompt = self.prompts_manager.build_starcoder_fix_prompt(
                    explanation=explanation,
                    refactored_code=current_code,
                    error_log=normalized_log,
                    focused_diff=focused_diff,
                    project_context=project_context,
                )

                fixed_code = perform_starcoder_refactor(
                    method_path=absolute_method,
                    method_name=method_name,
                    fix_prompt=starcoder_fix_prompt,
                    file_identifier=method_name,
                )

                iteration_info = {"iteration": iteration, "actions": []}
                iteration_info["actions"].append({
                    "step": "gpt_explanation",
                    "prompt": explanation_prompt,
                    "response": explanation,
                })

                if fixed_code.strip() == "cuda":
                    iteration_info["actions"].append({
                        "step": "starcoder_fix",
                        "prompt": starcoder_fix_prompt,
                        "response": "cuda",
                    })
                    iteration_info["actions"].append({
                        "step": "test_execution",
                        "pass": "cuda",
                        "output": "CUDA out of memory encountered during iterative StarCoder2 fix.",
                    })
                    result["logs"].append(iteration_info)
                    break

                with open(absolute_method, "w") as wf:
                    wf.write(fixed_code)
                current_code = fixed_code
                success, new_logs = self.run_single_test(pom_to_use, test_class)

                iteration_info["actions"].append({
                    "step": "starcoder_fix",
                    "prompt": starcoder_fix_prompt,
                    "response": fixed_code,
                })
                iteration_info["actions"].append({
                    "step": "test_execution",
                    "pass": success,
                    "output": new_logs,
                })
                result["logs"].append(iteration_info)

                self.short_term_memory.append({
                    "iteration": iteration,
                    "normalized_log": normalized_log,
                    "focused_diff": focused_diff,
                    "diagnosis": explanation,
                })

                iteration_file = os.path.join(refactored_methods_dir, f"{method_name}_iter{iteration}.java")
                with open(iteration_file, "w") as iter_file:
                    iter_file.write(fixed_code)

                if success:
                    result["success"] = True
                    result["iterations"] = iteration
                    result["final_code"] = current_code
                    break

                failure_logs = new_logs

            return result


        finally:
            pass

    ##########################
    # HELPER METHODS
    ##########################
    def strip_tmp_prefix(self, path: str) -> str:
        """
        Example: /tmp/repo_abc123/foo/bar => foo/bar
        """
        return re.sub(r"^/tmp/repo_[^/]+/", "", path)

    def derive_test_class_name(self, test_path: str) -> str:
        """
        E.g. restwars/service/src/test/java/com/example/MyTest.java => com.example.MyTest
        """
        if "/java/" in test_path:
            _, after_java = test_path.split("/java/", 1)
        else:
            after_java = test_path
        dotted = after_java.replace("/", ".")
        if dotted.endswith(".java"):
            dotted = dotted[:-5]
        return dotted

    def find_pom_for_test_file(self, test_file_rel_path: str) -> Optional[str]:
        """
        Walk upward from test_file_rel_path until we find a pom.xml or root.
        """
        test_file_abs_path = os.path.join(self.test_runner.local_repo_path, test_file_rel_path)
        if not os.path.exists(test_file_abs_path):
            return None

        current_dir = os.path.dirname(test_file_abs_path)
        while True:
            pom_candidate = os.path.join(current_dir, "pom.xml")
            if os.path.isfile(pom_candidate):
                return pom_candidate
            if os.path.abspath(current_dir) == os.path.abspath(self.test_runner.local_repo_path):
                break
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:
                break
            current_dir = parent_dir
        return None

    def run_single_test(self, pom_path: str, test_class_name: str) -> Tuple[bool, str]:
        """
        Runs 'mvn -f <pom_path> test -Dtest=<test_class_name>'
        Returns (pass_fail, logs)
        """
        cmd = ["mvn", "-f", pom_path, "test", f"-Dtest={test_class_name}"]
        print(f"[INFO] Running single test: {' '.join(cmd)}")

        import subprocess
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.test_runner.timeout_seconds
            )
            logs = proc.stdout + "\n" + proc.stderr
            passed = (proc.returncode == 0)
            return passed, logs
        except subprocess.TimeoutExpired:
            msg = f"[WARNING] Single test '{test_class_name}' timed out after {self.test_runner.timeout_seconds} seconds."
            return False, msg

    def _normalize_error_log(self, error_log: str) -> str:
        """
        Normalize environment-specific values so evidence is stable across machines.
        """
        clean = re.sub(r"\x1b\[[0-9;]*m", "", error_log)
        clean = re.sub(r"/tmp/repo_[^/\s]+", "<TMP_REPO>", clean)
        if self.test_runner.local_repo_path:
            clean = clean.replace(self.test_runner.local_repo_path, "<REPO_ROOT>")
        clean = re.sub(r"/[^\s:]+/([^/\s:]+\.java)", r"<ABS_PATH>/\1", clean)
        return clean

    def _extract_failure_signals(self, normalized_log: str) -> Dict[str, List[str] or str]:
        signature_parts: List[str] = []
        locations: List[str] = []
        symbols: List[str] = []

        exception_match = re.search(r"\b([A-Za-z_][A-Za-z0-9_\.]*?(?:Exception|Error))\b", normalized_log)
        if exception_match:
            signature_parts.append(exception_match.group(1))

        assertion_match = re.search(r"(expected:.*|assertion.*failed.*)", normalized_log, re.IGNORECASE)
        if assertion_match:
            signature_parts.append(assertion_match.group(1).strip())

        for m in re.finditer(r"([A-Za-z0-9_./-]+\.java):\[(\d+),(\d+)\]", normalized_log):
            locations.append(f"{m.group(1)}:{m.group(2)}")
        for m in re.finditer(r"\(([^()]+\.java):(\d+)\)", normalized_log):
            locations.append(f"{m.group(1)}:{m.group(2)}")

        for m in re.finditer(r"symbol:\s*(?:class|method|variable)?\s*([A-Za-z_][A-Za-z0-9_]*)", normalized_log):
            symbols.append(m.group(1))

        signature = " | ".join(signature_parts) if signature_parts else "Unknown failure signature"
        return {
            "signature": signature,
            "locations": locations[:6],
            "symbols": list(dict.fromkeys(symbols))[:12],
        }

    def _collect_symbols(self, signals: Dict, original_code: str, current_code: str) -> List[str]:
        symbols = list(signals.get("symbols", []))

        # Add symbols touched by diff lines to guide context retrieval.
        diff = self._compute_diff(original_code, current_code)
        for line in diff.splitlines():
            if not (line.startswith("+") or line.startswith("-")):
                continue
            for candidate in re.findall(r"\b[A-Z][A-Za-z0-9_]*\b|\b[a-z][A-Za-z0-9_]{3,}\b", line):
                if candidate in {"public", "private", "class", "return", "static", "import", "package"}:
                    continue
                symbols.append(candidate)

        deduped = []
        seen = set()
        for sym in symbols:
            if sym in seen:
                continue
            seen.add(sym)
            deduped.append(sym)
            if len(deduped) >= 20:
                break
        return deduped

    def _format_project_context(self, snippets: List[Dict]) -> str:
        if not snippets:
            return ""
        parts = []
        for item in snippets:
            parts.append(
                f"Symbol: {item.get('symbol')}\n"
                f"File: {item.get('path')}:{item.get('line')}\n"
                f"Snippet:\n{item.get('snippet', '').strip()}"
            )
        return "\n\n---\n\n".join(parts)

    def _extract_focused_diff(self, diff_str: str, symbols: List[str], max_hunks: int = 4) -> str:
        if not diff_str.strip().startswith("---"):
            return diff_str

        lines = diff_str.splitlines()
        header = []
        hunks: List[List[str]] = []
        current: List[str] = []
        for line in lines:
            if line.startswith("@@"):
                if current:
                    hunks.append(current)
                current = [line]
            elif current:
                current.append(line)
            else:
                header.append(line)
        if current:
            hunks.append(current)

        if not hunks:
            return diff_str

        selected: List[List[str]] = []
        lowered_symbols = [s.lower() for s in symbols if s]
        for hunk in hunks:
            hunk_text = "\n".join(hunk).lower()
            if any(sym in hunk_text for sym in lowered_symbols):
                selected.append(hunk)
            if len(selected) >= max_hunks:
                break

        if not selected:
            selected = hunks[:max_hunks]

        output_lines = list(header)
        for hunk in selected:
            output_lines.extend(hunk)
        return "\n".join(output_lines)

    def _compute_diff(self, original: str, updated: str) -> str:
        """
        Computes a unified diff between the original and updated Java code,
        ensuring imports and class definitions are included.
        """

        # Split original and updated code into lines
        original_lines = original.splitlines()
        updated_lines = updated.splitlines()

        # Ensure we capture context lines (like imports and class headers)
        diff_lines = list(difflib.unified_diff(
            original_lines,
            updated_lines,
            fromfile="Original",
            tofile="Refactored",
            lineterm=""
        ))

        # If diff output is too short, return full updated code
        if len(diff_lines) < 5:
            return updated  # If diff is too minimal, show full refactored code

        return "\n".join(diff_lines)
    