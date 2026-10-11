import os

import httpx


MODEL_SERVER_URL = os.getenv(
    "BIOMISTRAL_SERVER_URL",
    "http://127.0.0.1:8081",
)

SYSTEM_PROMPT = """
You are a sleep wellness education assistant.

Provide practical, evidence-informed information about sleep habits
and sleep efficiency using clear, accessible language.

Safety guidelines:
- Do not diagnose medical conditions or prescribe medication.
- Do not present sleep-efficiency estimates as medical diagnoses.
- Do not claim that feature contributions or associations prove causation.
- For persistent, severe, or concerning symptoms, recommend consulting
  a qualified healthcare professional.
- If someone feels sleepy while driving or operating machinery, advise
  them to stop safely and not resume until alert.
"""


async def generate_chat_response(message: str) -> str:
    url = f"{MODEL_SERVER_URL.rstrip('/')}/v1/chat/completions"

    payload = {
        "model": "BioMistral-Sleep-Assistant-V3",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": message,
            },
        ],
        "temperature": 0.2,
        "max_tokens": 300,
    }

    timeout = httpx.Timeout(90.0, connect=5.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

    except httpx.TimeoutException as exc:
        raise RuntimeError(
            "The BioMistral server took too long to respond."
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"The BioMistral server returned HTTP {exc.response.status_code}."
        ) from exc

    except httpx.RequestError as exc:
        raise RuntimeError(
            "Could not connect to the BioMistral server. "
            "Check that it is running."
        ) from exc

    choices = data.get("choices", [])

    if not choices:
        raise RuntimeError(
            "The BioMistral server returned no response choices."
        )

    content = choices[0].get("message", {}).get("content", "")

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            "The BioMistral server returned an empty response."
        )

    return content.strip()
