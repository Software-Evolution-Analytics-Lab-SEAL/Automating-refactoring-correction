# starcoder_refactoring.py
import csv
import os
import re
import requests
import guidance

from prompts_manager import PromptsManager

# ------------------------------------------------------------------
# CONFIGURATION: Update these if your local server uses a different
#   URL or model name.
API_URL    = "http://localhost:11434/api/generate"
MODEL_NAME = "starcoder2:15b-instruct-v0.1-fp16"
# ------------------------------------------------------------------

refactored_code_directory = "./refactored_methods"
if not os.path.exists(refactored_code_directory):
    os.makedirs(refactored_code_directory)



def call_local_starcoder2(prompt: str, stream: bool = False, options: dict = None) -> str:
    """
    Sends a POST request to locally hosted StarCoder2 API.
    Returns the raw 'response' string (i.e., the models generated text).
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": stream
    }
    # Add additional generation options (e.g., max_new_tokens) here:
    if options:
        payload["options"] = options

    try:
        resp = requests.post(API_URL, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        # If your server returns something like {"error": "CUDA out of memory"}, catch it:
        err = data.get("error", "").lower()
        if "out of memory" in err or "cuda" in err:
            return "cuda"

        return data.get("response", "").strip()
    except requests.exceptions.HTTPError as http_err:
        # If the HTTP status code itself indicates OOM (e.g., 507 Insufficient Storage),
        # you can check here as well:
        if resp.status_code == 507 or "out of memory" in resp.text.lower():
            return "cuda"
        print(f"[ERROR] HTTP error: {http_err}")
        return ""
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API request failed: {e}")
        return ""


def refactor_code_with_starcoder2(method_code: str, custom_prompt: str = None, file_identifier: str = None) -> str:
    """
    Refactor Java code using StarCoder2 while ensuring clean output.
    """

    prompts_manager = PromptsManager()

    if custom_prompt:
        instruction = custom_prompt.strip()  # Use the fix prompt
    else:
        instruction = f"""
Refactor the following Java method while maintaining its original functionality.

### Existing Code:
{method_code}

### Instructions:
- **Fix only performance, readability, and maintainability issues**.
- **Do NOT modify the class name, method signatures, or add new methods.**
- **Do NOT add comments or explanation.**
- **Return ONLY valid, runnable Java code (no extra text).**
""".strip()
        
    if file_identifier:
        prompts_manager.log_prompt_step(file_identifier, "StarCoder2 Refactor Prompt", instruction)
    else:
        prompts_manager.log_step("StarCoder2 Refactor Prompt", instruction)

    raw_response = call_local_starcoder2(instruction, stream=False, options={
        "max_new_tokens": 4096,
        "num_return_sequences": 1,
        "num_beams": 5,
        "do_sample": False,
        "truncation": True,
        "eos_token_id": None,  # Use model's default EOS token
    })

    if raw_response.strip() == "cuda":
        return "cuda"
    
    if not raw_response:
        print("No response returned from StarCoder2 API; skipping method.")
        return ""
    
    # Extract the original package declaration from the method code
    # This is to ensure the refactored code retains the original package context.
    package_match = re.search(r"^\s*package\s+[\w\.]+;", method_code, re.MULTILINE)
    original_package = package_match.group(0).strip() if package_match else ""

    # Extract original import statements
    original_imports = "\n".join(re.findall(r"import\s+[\w\.\*]+;", method_code))

    # Extract the code within the triple backticks.
    code_match = re.search(r"```(?:java)?\s*(.*?)(?:\s*```|$)", raw_response, re.DOTALL)
    if code_match:
        cleaned_response = code_match.group(1).strip()
    else:
        # Fallback to the entire response if no code block is detected.
        cleaned_response = raw_response.strip()

    # Ensure the code has necessary imports
    if "import" not in cleaned_response and original_imports:
        cleaned_response = f"{original_imports}\n\n{cleaned_response}"

    # Ensure the package definition is included at the top
    if original_package and original_package not in cleaned_response:
        cleaned_response = f"{original_package}\n\n{cleaned_response}"

    return cleaned_response


def perform_starcoder_refactor(method_path: str, method_name: str, fix_prompt: str = None, file_identifier: str = None) -> str:
    """
    Reads a Java file, locates the method by name, and calls StarCoder2 to refactor it.
    Uses `fix_prompt` if provided; otherwise, defaults to a general refactoring prompt.
    """
    # Read the original Java file
    with open(method_path, 'r') as file:
        original_code = file.read()

    # Extract the method code
    if method_name in original_code:
        method_code = original_code

        print(file_identifier)

        # Refactor the method using StarCoder2
        refactored_method = refactor_code_with_starcoder2(method_code, fix_prompt, file_identifier)

        # Check immediately for a CUDA error and exit if encountered.
        if refactored_method.strip() == "cuda":
            print("CUDA out of memory encountered during StarCoder2 generation. Skipping method.")
            return "cuda"

        # Save the refactored method to a new file
        refactored_file_path = os.path.join(
            refactored_code_directory,
            os.path.basename(method_path)
        )
        with open(refactored_file_path, 'w') as refactored_file:
            refactored_file.write(original_code.replace(method_code, refactored_method))

        print(f"Successfully refactored method: {method_name} in {method_path}")
        return original_code.replace(method_code, refactored_method)
    else:
        print(f"Method {method_name} not found in {method_path}")
        return ""
