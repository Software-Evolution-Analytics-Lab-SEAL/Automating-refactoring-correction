import os
from openai import OpenAI

class GPT4Wrapper:
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)  
        self.model_name = model_name

    def call_gpt4(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        """
        Sends a prompt to GPT-4 and returns the completion text.
        """
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
