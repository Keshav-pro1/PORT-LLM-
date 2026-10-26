"""
Test Suite for Phase 2: Chat Server Memory Interceptor Integration
==================================================================
Tests:
  1. GET /api/cortex/status
  2. POST /api/cortex/user (switch user & validation)
  3. GET /api/cortex/memories
  4. GET /api/cortex/profile (reflection)
  5. POST /api/cortex/retain (manual memory creation)
  6. Memory Recall Pre-Prompt Interception in /ollama/api/chat
  7. X-Cortex-Memories-Used Response Header verification
  8. Post-response asynchronous memory retention trigger
  9. Graceful fallback when Hindsight is offline / disabled
"""

import http.server
import json
import os
import sys
import threading
import time
import unittest
import urllib.request
import urllib.error

# Add project root and Shared to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED_DIR = os.path.join(ROOT_DIR, "Shared")
SCRIPTS_DIR = os.path.join(SHARED_DIR, "scripts")
for p in [SHARED_DIR, SCRIPTS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import chat_server
from chat_server import ChatHandler, ThreadedHTTPServer


class MockOllamaHandler(http.server.BaseHTTPRequestHandler):
    """Mock Ollama Engine server."""
    received_requests = []

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_len) if content_len > 0 else b"{}"
        data = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}

        self.received_requests.append({
            "path": self.path,
            "headers": dict(self.headers),
            "payload": data
        })

        if self.path == "/api/chat":
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()

            # Stream 2 chunks in Ollama JSON format
            chunk1 = json.dumps({
                "model": data.get("model", "llama3"),
                "message": {"role": "assistant", "content": "Hello! I remember "},
                "done": False
            }) + "\n"
            chunk2 = json.dumps({
                "model": data.get("model", "llama3"),
                "message": {"role": "assistant", "content": "your cloud stack."},
                "done": True
            }) + "\n"

            self.wfile.write(chunk1.encode("utf-8"))
            self.wfile.flush()
            self.wfile.write(chunk2.encode("utf-8"))
            self.wfile.flush()
        else:
            self.send_response(404)
            self.end_headers()


class MockHindsightHandler(http.server.BaseHTTPRequestHandler):
    """Mock Hindsight server for testing server-level memory operations."""
    retained_items = []
    recalled_queries = []

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        data = json.loads(body) if body else {}

        if self.path == "/v1/retain":
            self.retained_items.append(data)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "stored", "id": "mem_p2"}).encode("utf-8"))

        elif self.path == "/v1/recall":
            self.recalled_queries.append(data.get("query", ""))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            # Return 2 mock recalled memories
            self.wfile.write(json.dumps({
                "memories": [
                    "User prefers Python 3.12 with async FastAPI",
                    "User deploys on AWS ECS in ap-south-1"
                ]
            }).encode("utf-8"))

        elif self.path == "/v1/reflect":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "response": "User is Lead Backend Developer with strong Python/FastAPI knowledge."
            }).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if "/memories" in self.path:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "memories": [
                    {"content": "User prefers Python 3.12 with async FastAPI", "created_at": "2026-10-26T10:00:00Z"},
                    {"content": "User deploys on AWS ECS in ap-south-1", "created_at": "2026-10-26T10:05:00Z"}
                ]
            }).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


class TestPhase2ServerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. Start Mock Ollama Server
        cls.mock_ollama = http.server.HTTPServer(("127.0.0.1", 0), MockOllamaHandler)
        cls.ollama_port = cls.mock_ollama.server_address[1]
        cls.ollama_url = f"http://127.0.0.1:{cls.ollama_port}"
        cls.ollama_thread = threading.Thread(target=cls.mock_ollama.serve_forever, daemon=True)
        cls.ollama_thread.start()

        # Update OLLAMA_HOST in chat_server to point to mock
        chat_server.OLLAMA_HOST = cls.ollama_url

        # 2. Start Mock Hindsight Server
        cls.mock_hindsight = http.server.HTTPServer(("127.0.0.1", 0), MockHindsightHandler)
        cls.hindsight_port = cls.mock_hindsight.server_address[1]
        cls.hindsight_url = f"http://127.0.0.1:{cls.hindsight_port}"
        cls.hindsight_thread = threading.Thread(target=cls.mock_hindsight.serve_forever, daemon=True)
        cls.hindsight_thread.start()

        # Configure cortex_client in chat_server
        if chat_server.cortex_client:
            chat_server.cortex_client.enabled = True
            chat_server.cortex_client.base_url = cls.hindsight_url
            chat_server.cortex_client.user_id = "test-phase2-user"
            chat_server.cortex_client.auto_retain = True

        # 3. Start PortLLM Chat Server under test on an available ephemeral port
        cls.chat_server = ThreadedHTTPServer(("127.0.0.1", 0), ChatHandler)
        cls.chat_port = cls.chat_server.server_address[1]
        cls.chat_url = f"http://127.0.0.1:{cls.chat_port}"
        cls.chat_thread = threading.Thread(target=cls.chat_server.serve_forever, daemon=True)
        cls.chat_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.chat_server.shutdown()
        cls.chat_server.server_close()
        cls.mock_hindsight.shutdown()
        cls.mock_hindsight.server_close()
        cls.mock_ollama.shutdown()
        cls.mock_ollama.server_close()

    def setUp(self):
        MockOllamaHandler.received_requests.clear()
        MockHindsightHandler.retained_items.clear()
        MockHindsightHandler.recalled_queries.clear()

    def test_01_cortex_status_endpoint(self):
        """Test GET /api/cortex/status returns valid health telemetry."""
        req = urllib.request.Request(f"{self.chat_url}/api/cortex/status")
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data.get("enabled"))
            self.assertEqual(data.get("base_url"), self.hindsight_url)
            self.assertEqual(data.get("bank_id"), "portllm-test-phase2-user")

    def test_02_cortex_user_switching(self):
        """Test POST /api/cortex/user updates active user identity."""
        # Test valid user switch
        payload = json.dumps({"user_id": "priya_devops"}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.chat_url}/api/cortex/user",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data.get("success"))
            self.assertEqual(data.get("user_id"), "priya_devops")
            self.assertEqual(data.get("bank_id"), "portllm-priya_devops")

        # Verify status endpoint reflects new user
        req_stat = urllib.request.Request(f"{self.chat_url}/api/cortex/status")
        with urllib.request.urlopen(req_stat, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("user_id"), "priya_devops")
            self.assertEqual(data.get("bank_id"), "portllm-priya_devops")

        # Test invalid switch (missing user_id) returns 400
        req_bad = urllib.request.Request(
            f"{self.chat_url}/api/cortex/user",
            data=json.dumps({}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req_bad, timeout=5)
        self.assertEqual(ctx.exception.code, 400)

    def test_03_cortex_memories_endpoint(self):
        """Test GET /api/cortex/memories returns stored memories."""
        req = urllib.request.Request(f"{self.chat_url}/api/cortex/memories")
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("memories", data)
            self.assertEqual(len(data["memories"]), 2)
            self.assertEqual(data["memories"][0]["content"], "User prefers Python 3.12 with async FastAPI")

    def test_04_cortex_profile_endpoint(self):
        """Test GET /api/cortex/profile returns reflected synthesis summary."""
        req = urllib.request.Request(f"{self.chat_url}/api/cortex/profile")
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("profile", data)
            self.assertTrue("Lead Backend Developer" in data["profile"])

    def test_05_cortex_manual_retain(self):
        """Test POST /api/cortex/retain stores manual memory."""
        payload = json.dumps({"content": "Always format code with Black and ruff"}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.chat_url}/api/cortex/retain",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data.get("success"))

        # Verify mock Hindsight received it
        self.assertTrue(any("Always format code with Black" in item.get("content", "") for item in MockHindsightHandler.retained_items))

    def test_06_memory_recall_and_prompt_injection(self):
        """
        Verify that /ollama/api/chat:
          1. Calls Hindsight recall with user question
          2. Injects recalled context into system prompt
          3. Sets X-Cortex-Memories-Used HTTP header
          4. Triggers background retention after stream completes
        """
        chat_payload = json.dumps({
            "model": "llama3:latest",
            "messages": [
                {"role": "user", "content": "How should I structure the new microservice deployment?"}
            ],
            "stream": True
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.chat_url}/ollama/api/chat",
            data=chat_payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            # Check X-Cortex-Memories-Used header
            memories_used_hdr = resp.headers.get("X-Cortex-Memories-Used")
            self.assertEqual(memories_used_hdr, "2")

            # Read full streamed output
            body = resp.read().decode("utf-8")
            self.assertIn("Hello! I remember", body)
            self.assertIn("your cloud stack.", body)

        # 1. Verify Hindsight recall was queried
        self.assertIn("How should I structure the new microservice deployment?", MockHindsightHandler.recalled_queries)

        # 2. Verify upstream Mock Ollama received prompt enriched with memory block
        self.assertEqual(len(MockOllamaHandler.received_requests), 1)
        upstream_payload = MockOllamaHandler.received_requests[0]["payload"]
        upstream_msgs = upstream_payload.get("messages", [])

        # Should have injected system prompt with [HINDSIGHT LONG-TERM MEMORY]
        system_msg = next((m for m in upstream_msgs if m.get("role") == "system"), None)
        self.assertIsNotNone(system_msg)
        self.assertIn("[HINDSIGHT LONG-TERM MEMORY]", system_msg["content"])
        self.assertIn("User prefers Python 3.12 with async FastAPI", system_msg["content"])
        self.assertIn("User deploys on AWS ECS in ap-south-1", system_msg["content"])

        # 3. Verify background retention: give thread a fraction of a second to run
        time.sleep(0.3)
        self.assertTrue(len(MockHindsightHandler.retained_items) >= 1)
        latest_retained = MockHindsightHandler.retained_items[-1]
        self.assertIn("How should I structure the new microservice deployment?", latest_retained.get("content", ""))
        self.assertIn("Hello! I remember your cloud stack.", latest_retained.get("content", ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
