from pathlib import Path

import yaml
from ollama import chat


def load_model_config() -> dict:
    """Load the model configuration from models.yaml."""

    project_root = Path(__file__).resolve().parents[2]

    config_path = project_root / "configs" / "models.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def generate_response(
    prompt: str,
    system_prompt: str = "You are a helpful AI assistant."
) -> str:
    """Send a prompt to the configured Ollama model."""

    config = load_model_config()

    model_name = config["llm"]["model"]
    temperature = config["llm"]["temperature"]

    try:
        response = chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": temperature,
            },
        )

        return response.message.content

    except Exception as error:
        raise RuntimeError(
            f"Failed to generate a response using model '{model_name}': {error}"
        ) from error


if __name__ == "__main__":

    system_prompt = """
You are a biomedical research assistant.

Provide clear and concise answers.
Do not invent scientific evidence or citations.
If you are uncertain, clearly say that you are uncertain.
"""

    prompt = """
Explain Retrieval-Augmented Generation in two simple sentences.
"""

    answer = generate_response(
        prompt=prompt,
        system_prompt=system_prompt,
    )

    print("\nModel Response:\n")
    print(answer)