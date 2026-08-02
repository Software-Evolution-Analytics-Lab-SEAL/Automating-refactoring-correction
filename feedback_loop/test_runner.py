import os
import re
import shutil
import subprocess
import time
from typing import Optional, Tuple

class JavaTestRunner:
    """
    Clones/updates a Java Maven repo, builds it, runs tests (entire suite or single class),
    and cleans up.
    """

    def __init__(
        self,
        repo_name: str,
        repo_url: str,
        clone_dir: str = "./cloned_repos",
        timeout_seconds: int = 600
    ):
        self.repo_name = repo_name
        self.repo_url = repo_url
        self.clone_dir = clone_dir
        self.timeout_seconds = timeout_seconds

        self.local_repo_path = os.path.join(self.clone_dir, repo_name.replace("/", "_"))
    

    def clone_or_update_repo(self) -> None:
        if not os.path.exists(self.local_repo_path):
            print(f"[INFO] Cloning (recursive) {self.repo_url} into {self.local_repo_path} ...")
            result = subprocess.run(
                ["git", "clone", "--recursive", self.repo_url, self.local_repo_path],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"[ERROR] Cloning failed: {result.stderr}")
            else:
                print(f"[INFO] Cloning completed successfully.")
        else:
            print(f"[INFO] Repo already exists at {self.local_repo_path}. Resetting repository to a clean state...")
            try:
                # Discard any local changes
                subprocess.run(["git", "-C", self.local_repo_path, "reset", "--hard", "HEAD"], check=True)
                # Remove untracked files and directories
                subprocess.run(["git", "-C", self.local_repo_path, "clean", "-fdx"], check=True)
                # Pull the latest changes
                subprocess.run(["git", "-C", self.local_repo_path, "pull"], check=False)
                # Update submodules
                subprocess.run(["git", "-C", self.local_repo_path, "submodule", "update", "--init", "--recursive"], check=False)
                print(f"[INFO] Repository reset and updated successfully.")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to reset repository: {e}")

        
    def find_top_level_pom(self) -> Optional[str]:
        candidate = os.path.join(self.local_repo_path, "pom.xml")
        return candidate if os.path.isfile(candidate) else None

    def build_entire_project(self, pom_path: str) -> bool:
        if not os.path.isfile(pom_path):
            return False
        print(f"[INFO] Building aggregator: {pom_path} (skipTests)")
        cmd = ["mvn", "-f", pom_path, "clean", "install", "-DskipTests"]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
            print(result.stdout, result.stderr)
            return (result.returncode == 0)
        except subprocess.TimeoutExpired:
            print(f"[WARNING] Build timed out after {self.timeout_seconds} secs.")
            return False

    def run_maven_tests(self, pom_path: str, test_class_name: Optional[str] = None) -> Tuple[bool, str]:
        """
        Runs the entire suite if test_class_name is None;
        runs a single test class if test_class_name is provided -> -Dtest=<test_class_name>.
        Returns (success, logs).
        """
        cmd = ["mvn", "-f", pom_path, "test"]
        if test_class_name:
            cmd.append(f"-Dtest={test_class_name}")

        print(f"[INFO] Running command: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
            logs = proc.stdout + "\n" + proc.stderr
            success = (proc.returncode == 0)
            return success, logs
        except subprocess.TimeoutExpired:
            msg = f"[WARNING] Tests timed out after {self.timeout_seconds} seconds."
            return False, msg


    @staticmethod
    def strip_tmp_prefix(path: str) -> str:
        """
        Example: /tmp/repo_abc123/foo/bar => foo/bar
        """
        return re.sub(r"^/tmp/repo_[^/]+/", "", path)

    @staticmethod
    def derive_test_class_name(test_path: str) -> str:
        """
        E.g. restwars/service/src/test/java/com/example/MyTest.java
             => com.example.MyTest
        Splits at '/java/' and replaces '/' with '.' plus removing '.java'
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
        Start from the directory containing the test file, walk upward until
        we find 'pom.xml' or reach local_repo_path. Return path or None.
        """
        test_file_abs_path = os.path.join(self.local_repo_path, test_file_rel_path)
        if not os.path.exists(test_file_abs_path):
            return None

        current_dir = os.path.dirname(test_file_abs_path)
        while True:
            pom_candidate = os.path.join(current_dir, "pom.xml")
            if os.path.isfile(pom_candidate):
                return pom_candidate
            if os.path.abspath(current_dir) == os.path.abspath(self.local_repo_path):
                break
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:
                break
            current_dir = parent_dir
        return None

    def run_single_test(self, pom_path: str, test_class_name: str) -> bool:
        """
        e.g. mvn -f <pom_path> test -Dtest=<test_class_name>
        Returns True if the test passes, otherwise False.
        """
        cmd = ["mvn", "-f", pom_path, "test", f"-Dtest={test_class_name}"]
        print(f"[INFO] Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
            print(result.stdout)
            print(result.stderr)
            return (result.returncode == 0)
        except subprocess.TimeoutExpired:
            print(f"[WARNING] Test '{test_class_name}' timed out after {self.timeout_seconds} seconds. Marking test as failed.")
            return False
