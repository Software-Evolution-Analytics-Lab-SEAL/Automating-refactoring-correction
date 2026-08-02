from typing import List, Dict
import os
import datetime
import re

class PromptsManager:
    """
    Builds prompts for GPT-4 and StarCoder2 that incorporate:
      - Original Code Snippet
      - Refactored Code Snippet or diff
      - Test logs / error messages
      - Retrieved knowledge-base entries
      - Self-reflective instructions
      
    Logs each step of the process into separate files for debugging and traceability.
    Two directories are created:
      - prompt_logs: for prompts (e.g. StarCoder2 fix prompts, GPT-4 explanation prompts)
      - test_results: for test logs and results.
    """
    def __init__(self, log_dir: str = "./logs"):
        self.system_prompt = """
You are a code refactoring assistant. Your task is to analyze Java code that has been previously refactored incorrectly, and provide explanations for issues.
"""
        # Ensure the main log directory exists.
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        # Create separate directories for prompt logs and test results.
        self.prompt_log_dir = os.path.join(self.log_dir, "prompt_logs")
        self.test_result_dir = os.path.join(self.log_dir, "test_results")
        os.makedirs(self.prompt_log_dir, exist_ok=True)
        os.makedirs(self.test_result_dir, exist_ok=True)

    def _get_timestamp(self) -> str:
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_prompt_step(self, file_identifier: str, step_name: str, content: str) -> None:
        """
        Logs a prompt-related step into a file inside the prompt_logs directory.
        The file name is based on the file_identifier.
        """
        filename = os.path.join(self.prompt_log_dir, f"{file_identifier}.txt")
        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== {step_name} ({self._get_timestamp()}) ===\n")
            f.write(content)
            f.write("\n")

    def log_test_step(self, file_identifier: str, step_name: str, content: str) -> None:
        """
        Logs a test result step into a file inside the test_results directory.
        The file name is based on the file_identifier.
        """
        filename = os.path.join(self.test_result_dir, f"{file_identifier}.txt")
        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== {step_name} ({self._get_timestamp()}) ===\n")
            f.write(content)
            f.write("\n")


    def _crop_error_log(self, error_log: str) -> str:
        """
        Strip ANSI colors, then extract only compilation errors of the form
        "[line,col] cannot find symbol" plus their immediate
        'symbol:' and 'location:' lines.
        """
        # 1) remove ANSI escape sequences
        clean = re.sub(r'\x1b\[[0-9;]*m', '', error_log)
        lines = clean.splitlines()
        cropped = []

        for i, line in enumerate(lines):
            # only match real compile‐errors
            # look for ANY compile‐error on a coordinate line
            m = re.search(r'\[(\d+,\d+)\]\s*(.*error.*|cannot.*|incompatible types:.*|package .* does not exist|; expected|illegal start).*', line, re.IGNORECASE)
            if m:
                coord, msg = m.groups()
                cropped.append(f'[{coord}] {msg.strip()}')
                # next two lines if they exist
                if i+1 < len(lines) and lines[i+1].strip().lower().startswith('symbol:'):
                    cropped.append(lines[i+1].strip())
                if i+2 < len(lines) and lines[i+2].strip().lower().startswith('location:'):
                    cropped.append(lines[i+2].strip())
                cropped.append("")  # blank separator

        # fallback if nothing found
        if not cropped:
            return "\n".join(l for l in lines if 'cannot find symbol' in l.lower()).strip()

        return "\n".join(cropped).strip()
    

    def build_explanation_prompt(
        self,
        refactored_diff: str,
        error_log: str,
        test_name: str,
        knowledge_base_entries: List[Dict],
        error_signature: str = "",
        failure_location: str = "",
        project_context: str = ""
    ) -> str:
        """
        Builds a prompt instructing GPT-4 to explain likely causes of failure.
        """
        kb_context = "\n".join([f"- {entry['common_fix']}" for entry in knowledge_base_entries])
        prompt = f"""
{self.system_prompt}

We have the following refactored code diff:

{refactored_diff}

Test '{test_name}' failed with the following error log:

{error_log}

    Error signature:
    {error_signature or 'N/A'}

    Approximate failure location:
    {failure_location or 'N/A'}

    Retrieved repository context:
    {project_context or 'N/A'}

Here are some common fixes or patterns that might be relevant:
{kb_context}

    Return exactly two labeled lines:
    Root Cause: <one concise sentence>
    Hint: <one concise sentence with concrete symbol names and target edits>
"""
        # Log the explanation prompt in a file named after the test.
        self.log_prompt_step(test_name, "Input Suggestion", prompt.strip())
        return prompt.strip()

    def build_starcoder_fix_prompt(
        self,
        explanation: str,
        refactored_code: str,
        error_log: str,
        focused_diff: str = "",
        project_context: str = ""
    ) -> str:
        """
        Builds a StarCoder2 fix prompt ensuring only valid Java code is returned.
        """
        short_error_log = self._crop_error_log(error_log)
        print(short_error_log)
        prompt = f"""
Refactor the following Java code to fix errors based on the given explanation and error log.

### Existing (broken) code:
{refactored_code}

### Unit test failed with this error log:
{short_error_log}

### Focused diff region:
{focused_diff or 'N/A'}

### Project context snippets:
{project_context or 'N/A'}

### Explanation of the issue:
{explanation}

### Instructions:
- **Do NOT modify class names or add extra classes.**
- **Do NOT generate explanations or markdown (```java```).**
- **Return ONLY clean, runnable Java code with NO additional text.**
- **Ensure class, method, and brace structures remain correct.**
"""
        # Log the StarCoder fix prompt.
        self.log_prompt_step("StarCoderFix", "Input StarCoder Fix", prompt.strip())
        return prompt.strip()
