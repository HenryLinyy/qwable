"""Model client wrappers for Ollama and ds4."""

import httpx
import json
import logging
import subprocess

logger = logging.getLogger("qwable.models")


def _completion_is_empty(result: dict) -> bool:
    """True if a chat completion has blank assistant content and no tool_calls.

    Used to detect the cold-load empty-content case (thinking models emit nothing
    on their first call after loading). A tool-only response (empty content +
    tool_calls) is NOT considered empty.
    """
    try:
        choices = result.get("choices") or []
        if not choices:
            return False
        message = choices[0].get("message") or {}
        if message.get("tool_calls"):
            return False
        content = message.get("content")
        return not (content and str(content).strip())
    except Exception:
        return False


class OllamaClient:
    """HTTP client for local OpenAI-compatible model backends.

    The class name is kept for compatibility with the v1.5 codebase, but the
    backend can be Ollama or LM Studio. Chat completions use the shared
    OpenAI-compatible API; native multimodal/unload behavior is shimmed per
    backend.
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 900,
        backend: str = "ollama",
        lmstudio_cli_path: str | None = None,
        ttl_seconds: int | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.backend = backend
        self.lmstudio_cli_path = lmstudio_cli_path
        self.ttl_seconds = ttl_seconds
        self.client = httpx.Client(timeout=timeout)

    def _is_lmstudio(self) -> bool:
        return self.backend.lower() == "lmstudio"

    def _apply_ttl(self, payload: dict) -> dict:
        """Add LM Studio's JIT auto-unload TTL so idle models free memory faster
        than the 1h default. Ignored by non-LM-Studio backends."""
        if self._is_lmstudio() and self.ttl_seconds and self.ttl_seconds > 0:
            payload.setdefault("ttl", self.ttl_seconds)
        return payload

    def chat_completion(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 1200,
        stream: bool = False,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> dict:
        """Call Ollama /v1/chat/completions. Returns the full response."""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
        self._apply_ttl(payload)

        result = self._post_chat(payload)
        # Thinking models (gemma-4, qwen3.6, Qwable) frequently return EMPTY
        # content on the first call right after the model loads (cold start).
        # Retry once — by then the model is warm and emits real content. Skip the
        # retry for streaming and for legitimate tool-only responses.
        if not stream and _completion_is_empty(result):
            logger.info(
                "empty completion from %s — retrying once (likely cold-load)", model
            )
            result = self._post_chat(payload)
        return result

    def _post_chat(self, payload: dict) -> dict:
        response = self.client.post(f"{self.base_url}/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()

    def chat_completion_stream(
        self,
        *,
        model: str,
        messages: list[dict],
        max_tokens: int = 1200,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
    ):
        """Sync generator yielding (delta_text, finish_reason) per SSE chunk.

        Used by G11 fusion streaming runner to feed judge token-by-token
        into FusionStreamEvent stream. Stops at `data: [DONE]` or terminal
        finish_reason.

        Yields:
            tuple[str, str | None]: (delta_text, finish_reason)
                - delta_text: incremental content from this chunk ("")
                - finish_reason: "stop" / "length" / None (in-progress)
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
        self._apply_ttl(payload)

        with self.client.stream(
            "POST", f"{self.base_url}/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                # iter_lines yields bytes (when decode_unicode=False) or str
                line = (
                    raw_line.decode("utf-8")
                    if isinstance(raw_line, bytes)
                    else raw_line
                )
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: ") :]
                if data_str.strip() == "[DONE]":
                    return
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                try:
                    choice = data["choices"][0]
                except (KeyError, IndexError, TypeError):
                    continue
                delta = (choice.get("delta") or {}).get("content") or ""
                finish = choice.get("finish_reason")
                yield (delta, finish)

    def native_chat_completion(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 1200,
        stream: bool = False,
        temperature: float = 0.0,
        think: bool | None = None,
    ) -> dict:
        """Call the backend's multimodal chat API and return native-like output."""
        if self._is_lmstudio():
            payload = {
                "model": model,
                "messages": self._lmstudio_messages_from_native(messages),
                "max_tokens": max_tokens,
                "stream": stream,
                "temperature": temperature,
            }
            self._apply_ttl(payload)
            response = self.client.post(
                f"{self.base_url}/chat/completions", json=payload
            )
            response.raise_for_status()
            data = response.json()
            message = data.get("choices", [{}])[0].get("message", {})
            return {"message": message}

        native_base_url = (
            self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        )
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if think is not None:
            payload["think"] = think

        response = self.client.post(f"{native_base_url}/api/chat", json=payload)
        response.raise_for_status()
        return response.json()

    def _lmstudio_messages_from_native(self, messages: list[dict]) -> list[dict]:
        """Convert Ollama native image messages to OpenAI multimodal messages."""
        converted = []
        for message in messages:
            new_message = {k: v for k, v in message.items() if k != "images"}
            images = message.get("images") or []
            if images:
                content_parts = []
                text = message.get("content")
                if text:
                    content_parts.append({"type": "text", "text": text})
                for image_b64 in images:
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        }
                    )
                new_message["content"] = content_parts
            converted.append(new_message)
        return converted

    def unload_models(self, models: list[str]) -> None:
        """Release resident local backend models before ds4-heavy work."""
        if self._is_lmstudio():
            if not self.lmstudio_cli_path:
                logger.info("LM Studio unload skipped: no lmstudio_cli_path configured")
                return
            try:
                subprocess.run(
                    [self.lmstudio_cli_path, "unload", "--all"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except Exception as e:
                logger.info("LM Studio unload skipped: %s", e)
            return

        native_base_url = (
            self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        )
        seen = set()
        for model in models:
            if not model or model in seen:
                continue
            seen.add(model)
            try:
                response = self.client.post(
                    f"{native_base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": "",
                        "stream": False,
                        "keep_alive": 0,
                    },
                )
                response.raise_for_status()
            except Exception as e:
                logger.info("Ollama unload skipped for %s: %s", model, e)

    def close(self):
        self.client.close()


class DS4Client:
    """HTTP client for ds4-server."""

    def __init__(self, base_url: str, timeout: int = 1200):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def chat_completion(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 3600,
        stream: bool = False,
        temperature: float = 0.7,
    ) -> dict:
        """Call ds4 /v1/chat/completions. Returns the full response."""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
            "temperature": temperature,
        }
        response = self.client.post(f"{self.base_url}/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()

    def health(self) -> bool:
        """Check if ds4 server is alive."""
        try:
            response = self.client.get(f"{self.base_url}/models")
            if response.status_code != 200:
                return False
            data = response.json()
            return data.get("object") == "list" and isinstance(data.get("data"), list)
        except Exception:
            return False

    def close(self):
        self.client.close()
