"""
Test Suite for Phase 1: Zero-Dependency Hindsight Client
=========================================================
Tests:
  1. Config loading & defaults
  2. Per-user bank namespace normalization
  3. Thread-safe user identity switching
  4. Offline / unreachable API resilience (zero-crash fallback)
  5. Mock Hindsight HTTP server retain, recall, reflect, and list operations
  6. Status dictionary contract
"""

import http.server
import json
import os
import sys
import tempfile
import threading
import unittest

# Add project root and Shared/scripts to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "Shared", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from hindsight_client import HindsightClient


class MockHindsightHandler(http.server.BaseHTTPRequestHandler):
    """Mock Hindsight API server for offline integration testing."""
    received_requests = []
    memories_store = {}

    def log_message(self, format, *args):
        pass  # Quiet logger during unit tests

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        data = json.loads(post_body) if post_body else {}

        auth_header = self.headers.get("Authorization", "")
        self.received_requests.append({
            "method": "POST",
            "path": self.path,
            "auth": auth_header,
            "body": data
        })

        # Extract bank_id from path if /v1/default/banks/{bank_id}/...
        bank = "default"
        if "/banks/" in self.path:
            parts = self.path.split("/")
            b_idx = parts.index("banks")
            if len(parts) > b_idx + 1:
                bank = parts[b_idx + 1]
        else:
            bank = data.get("bank_id", "default")

        if self.path.endswith("/memories") or self.path == "/v1/retain":
            items = data.get("items", [])
            if items and isinstance(items, list):
                content = items[0].get("content", "")
            else:
                content = data.get("content", "")
            if bank not in self.memories_store:
                self.memories_store[bank] = []
            self.memories_store[bank].append({"content": content, "created_at": "2026-10-26T12:00:00Z"})

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "status": "retained", "id": "mem_123"}).encode("utf-8"))

        elif self.path.endswith("/memories/recall") or self.path == "/v1/recall":
            stored = self.memories_store.get(bank, [])
            results = [m["content"] for m in stored]

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"results": [{"text": m["content"]} for m in stored], "memories": results}).encode("utf-8"))

        elif self.path.endswith("/reflect") or self.path == "/v1/reflect":
            stored = self.memories_store.get(bank, [])
            count = len(stored)
            msg = f"Profile synthesis for {bank}: user has {count} active memories."

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "text": msg,
                "response": msg
            }).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        self.received_requests.append({
            "method": "GET",
            "path": self.path,
            "auth": self.headers.get("Authorization", "")
        })

        if "/memories" in self.path:
            bank = "default"
            if "/banks/" in self.path:
                parts = self.path.split("/")
                b_idx = parts.index("banks")
                if len(parts) > b_idx + 1:
                    bank = parts[b_idx + 1]
            stored = self.memories_store.get(bank, [])

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"items": [{"text": m["content"], "date": m["created_at"]} for m in stored], "memories": stored}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


class TestHindsightClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start mock HTTP server on an available port
        cls.mock_server = http.server.HTTPServer(("127.0.0.1", 0), MockHindsightHandler)
        cls.mock_port = cls.mock_server.server_address[1]
        cls.mock_url = f"http://127.0.0.1:{cls.mock_port}"
        cls.server_thread = threading.Thread(target=cls.mock_server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.mock_server.shutdown()
        cls.mock_server.server_close()

    def setUp(self):
        MockHindsightHandler.received_requests.clear()
        MockHindsightHandler.memories_store.clear()

    def test_01_config_loading_and_bank_namespacing(self):
        """Test config JSON parsing and bank namespace normalization."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            json.dump({
                "hindsight_enabled": True,
                "hindsight_api_url": "https://api.hindsight.test",
                "hindsight_api_key": "test-key-123",
                "user_id": "Rahul DevOps Lead",
                "recall_top_k": 7
            }, f)
            cfg_path = f.name

        try:
            client = HindsightClient(config_path=cfg_path)
            self.assertTrue(client.enabled)
            self.assertEqual(client.base_url, "https://api.hindsight.test")
            self.assertEqual(client.api_key, "test-key-123")
            self.assertEqual(client.user_id, "Rahul DevOps Lead")
            self.assertEqual(client.bank_id, "portllm-rahul_devops_lead")
            self.assertEqual(client.recall_top_k, 7)
        finally:
            os.remove(cfg_path)

    def test_02_user_switching(self):
        """Test switching active user identities updates the memory namespace."""
        client = HindsightClient()
        client.user_id = "default"
        self.assertEqual(client.bank_id, "portllm-default")

        client.update_user("priya-backend")
        self.assertEqual(client.user_id, "priya-backend")
        self.assertEqual(client.bank_id, "portllm-priya-backend")

        client.update_user("Special!@#User$")
        self.assertEqual(client.bank_id, "portllm-special___user_")

    def test_03_offline_graceful_fallback(self):
        """Verify client never raises exceptions when API is offline or unconfigured."""
        # Unconfigured / disabled client
        client_disabled = HindsightClient()
        client_disabled.enabled = False

        self.assertIsNone(client_disabled.retain("Some knowledge"))
        self.assertEqual(client_disabled.recall("Query"), [])
        self.assertEqual(client_disabled.reflect("Profile query"), "")
        self.assertEqual(client_disabled.list_memories(), [])

        # Enabled but unreachable server (non-existent port)
        client_unreachable = HindsightClient()
        client_unreachable.enabled = True
        client_unreachable.base_url = "http://127.0.0.1:59999"  # Dead port
        client_unreachable.timeout = 1

        self.assertIsNone(client_unreachable.retain("Should not crash"))
        self.assertEqual(client_unreachable.recall("Should not crash"), [])
        self.assertEqual(client_unreachable.reflect("Should not crash"), "")
        self.assertEqual(client_unreachable.list_memories(), [])

    def test_04_mock_server_retain_recall_reflect_cycle(self):
        """Verify full retain -> recall -> reflect -> list_memories cycle with mock server."""
        client = HindsightClient()
        client.enabled = True
        client.base_url = self.mock_url
        client.api_key = "test-secret-token"
        client.update_user("rahul")

        # 1. Retain knowledge
        ret1 = client.retain("User is Senior Backend Engineer working with FastAPI and PostgreSQL.")
        self.assertIsNotNone(ret1)
        self.assertEqual(ret1.get("status"), "retained")

        ret2 = client.retain("Prefers Docker Compose and Makefile workflows over GUI.")
        self.assertIsNotNone(ret2)

        # 2. Recall knowledge
        memories = client.recall("How does user deploy services?")
        self.assertEqual(len(memories), 2)
        self.assertIn("User is Senior Backend Engineer working with FastAPI and PostgreSQL.", memories)
        self.assertIn("Prefers Docker Compose and Makefile workflows over GUI.", memories)

        # 3. List memories
        all_memories = client.list_memories()
        self.assertEqual(len(all_memories), 2)
        self.assertEqual(all_memories[0]["content"], "User is Senior Backend Engineer working with FastAPI and PostgreSQL.")

        # 4. Reflect knowledge
        reflection = client.reflect("Summarize Rahul's profile")
        self.assertTrue("portllm-rahul" in reflection)
        self.assertTrue("2 active memories" in reflection)

        # Verify Authorization header was sent
        for req in MockHindsightHandler.received_requests:
            self.assertEqual(req["auth"], "Bearer test-secret-token")

    def test_05_multi_user_isolation(self):
        """Verify memory banks are strictly isolated between different users."""
        client = HindsightClient()
        client.enabled = True
        client.base_url = self.mock_url

        # Store memory for User A
        client.update_user("user-a")
        client.retain("User A secret memory: Django on AWS")

        # Store memory for User B
        client.update_user("user-b")
        client.retain("User B secret memory: Go on Kubernetes")

        # Recall for User B
        mem_b = client.recall("What is user stack?")
        self.assertEqual(len(mem_b), 1)
        self.assertIn("User B secret memory: Go on Kubernetes", mem_b)
        self.assertNotIn("User A secret memory: Django on AWS", mem_b)

        # Recall for User A
        client.update_user("user-a")
        mem_a = client.recall("What is user stack?")
        self.assertEqual(len(mem_a), 1)
        self.assertIn("User A secret memory: Django on AWS", mem_a)
        self.assertNotIn("User B secret memory: Go on Kubernetes", mem_a)

    def test_06_status_contract(self):
        """Verify get_status returns expected telemetry dictionary."""
        client = HindsightClient()
        client.enabled = True
        client.base_url = "http://test-server"
        client.api_key = "xyz"
        client.update_user("demo")

        status = client.get_status()
        self.assertTrue(status["enabled"])
        self.assertEqual(status["base_url"], "http://test-server")
        self.assertEqual(status["user_id"], "demo")
        self.assertEqual(status["bank_id"], "portllm-demo")
        self.assertTrue(status["has_api_key"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
