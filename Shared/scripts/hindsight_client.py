"""
PortLLM Cortex — Hindsight Memory Client
=========================================
Zero-dependency HTTP client for the Hindsight persistent memory API.
Uses only Python stdlib (urllib.request, json, os, logging, threading).

Features:
  - Supports live Hindsight Cloud & Docker API (/v1/default/banks/{bank_id}/*)
  - Per-user memory bank namespacing: portllm-{user_id}
  - Automatic identity recall before LLM prompt execution
  - Thread-safe background memory retention
  - Deep memory reflection for user profiling
  - Graceful fallback: 100% resilient if Hindsight service is offline
"""

import json
import logging
import os
import threading
import urllib.error
import urllib.request

logger = logging.getLogger("portllm.cortex")


class HindsightClient:
    """Lightweight, resilient Hindsight client using Python standard library."""

    def __init__(self, config_path=None):
        self.enabled = False
        self.base_url = ""
        self.api_key = ""
        self.user_id = "default"
        self.recall_top_k = 5
        self.auto_retain = True
        self.timeout = 15
        self.config_path = config_path
        self._lock = threading.Lock()

        # If no config_path passed, locate standard config path
        if not self.config_path:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.normpath(os.path.join(script_dir, "..", "config", "hindsight_config.json"))
            if os.path.isfile(candidate):
                self.config_path = candidate

        if self.config_path and os.path.isfile(self.config_path):
            self._load_config(self.config_path)

    def _load_config(self, path):
        """Safely load Hindsight configuration from a JSON file."""
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                cfg = json.load(f)
            self.enabled = bool(cfg.get("hindsight_enabled", False))
            self.base_url = str(cfg.get("hindsight_api_url", "")).rstrip("/")
            self.api_key = str(cfg.get("hindsight_api_key", "")).strip()
            self.user_id = str(cfg.get("user_id", "default")).strip() or "default"
            self.recall_top_k = int(cfg.get("recall_top_k", 5))
            self.auto_retain = bool(cfg.get("auto_retain", True))
            self.timeout = int(cfg.get("timeout_seconds", 15))

            if self.enabled and self.base_url:
                logger.info("Hindsight Cortex initialized: %s (Bank: %s)", self.base_url, self.bank_id)
            else:
                logger.info("Hindsight Cortex disabled or URL unconfigured.")
        except Exception as e:
            logger.warning("Failed to load Hindsight config from %s: %s", path, e)
            self.enabled = False

    @property
    def bank_id(self):
        """Per-user memory bank namespace."""
        safe_user = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in self.user_id).lower()
        return f"portllm-{safe_user}"

    def update_user(self, user_id):
        """Thread-safe switch of active user identity / bank namespace."""
        if not user_id:
            return self.user_id
        with self._lock:
            self.user_id = str(user_id).strip() or "default"
            logger.info("Cortex user namespace switched to: %s (bank: %s)", self.user_id, self.bank_id)
        return self.user_id

    def _request(self, method, path, body=None, timeout=None):
        """Execute HTTP request to Hindsight API with error handling and fallback."""
        if not self.enabled or not self.base_url:
            return None

        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        req_timeout = timeout or self.timeout

        try:
            with urllib.request.urlopen(req, timeout=req_timeout) as resp:
                resp_bytes = resp.read()
                if not resp_bytes:
                    return {}
                return json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                err_body = ""
            logger.error("Hindsight HTTP error %d [%s]: %s", e.code, url, err_body)
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            logger.warning("Hindsight connection failed [%s]: %s", url, e)
            return None
        except Exception as e:
            logger.error("Unexpected error contacting Hindsight [%s]: %s", url, e)
            return None

    def retain(self, content, metadata=None):
        """
        Store knowledge in active user's memory bank.
        Called asynchronously after an AI response completes.
        """
        if not content or not str(content).strip():
            return None

        text_content = str(content).strip()

        # Primary: Official Hindsight OpenAPI endpoint
        body_v1 = {
            "async": False,
            "items": [{"content": text_content}]
        }
        res = self._request("POST", f"/v1/default/banks/{self.bank_id}/memories", body=body_v1)
        if res is not None:
            return res

        # Fallback / Mock server endpoint
        body_fallback = {
            "bank_id": self.bank_id,
            "content": text_content
        }
        if metadata and isinstance(metadata, dict):
            body_fallback["metadata"] = metadata
        return self._request("POST", "/v1/retain", body=body_fallback)

    def recall(self, query, top_k=None):
        """
        Retrieve relevant memories before generating an LLM response.
        Returns a list of retrieved memory strings.
        """
        if not query or not str(query).strip():
            return []

        q = str(query).strip()

        # Primary: Official Hindsight OpenAPI endpoint
        body_v1 = {"query": q}
        res = self._request("POST", f"/v1/default/banks/{self.bank_id}/memories/recall", body=body_v1)

        # Fallback / Mock server endpoint
        if res is None:
            body_fallback = {
                "bank_id": self.bank_id,
                "query": q,
                "top_k": top_k or self.recall_top_k
            }
            res = self._request("POST", "/v1/recall", body=body_fallback)

        if not res:
            return []

        memories = []
        raw_items = res.get("results") or res.get("memories") or []
        if isinstance(raw_items, list):
            for item in raw_items:
                if isinstance(item, str) and item.strip():
                    memories.append(item.strip())
                elif isinstance(item, dict):
                    content = item.get("text") or item.get("content") or item.get("memory")
                    if content:
                        memories.append(str(content).strip())
        return memories

    def reflect(self, query="Synthesize what is known about this user, their role, stack, and preferences."):
        """
        Deep memory synthesis — ask Hindsight to reflect across memories.
        Used for generating user profile cards and persona summaries.
        """
        q = str(query).strip()

        # Primary: Official Hindsight OpenAPI endpoint
        body_v1 = {"query": q}
        res = self._request("POST", f"/v1/default/banks/{self.bank_id}/reflect", body=body_v1)

        # Fallback / Mock server endpoint
        if res is None:
            body_fallback = {
                "bank_id": self.bank_id,
                "query": q
            }
            res = self._request("POST", "/v1/reflect", body=body_fallback)

        if not res:
            return ""

        return str(res.get("text") or res.get("response") or res.get("reflection") or res.get("summary") or "").strip()

    def list_memories(self):
        """
        List all memories in active user's bank for the UI inspector.
        Returns a list of dicts: [{'content': ..., 'created_at': ...}, ...]
        """
        # Primary: Official Hindsight OpenAPI endpoint
        res = self._request("GET", f"/v1/default/banks/{self.bank_id}/memories/list")

        # Fallback / Mock server endpoint
        if res is None:
            res = self._request("GET", f"/v1/banks/{self.bank_id}/memories")

        if not res:
            return []

        raw_items = res.get("items") or res.get("memories") or []
        formatted = []
        if isinstance(raw_items, list):
            for item in raw_items:
                if isinstance(item, str):
                    formatted.append({"content": item, "created_at": None})
                elif isinstance(item, dict):
                    formatted.append({
                        "content": item.get("text") or item.get("content") or item.get("memory") or "",
                        "created_at": item.get("date") or item.get("created_at") or item.get("timestamp")
                    })
        return formatted

    def get_status(self):
        """Return connectivity and configuration status for the UI & diagnostics."""
        return {
            "enabled": self.enabled,
            "base_url": self.base_url,
            "user_id": self.user_id,
            "bank_id": self.bank_id,
            "has_api_key": bool(self.api_key),
            "recall_top_k": self.recall_top_k,
            "auto_retain": self.auto_retain
        }
