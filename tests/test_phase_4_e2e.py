"""
Test Suite for Phase 4: End-to-End Multi-Turn & Multi-User Verification
========================================================================
Simulates real-world enterprise scenarios:
  1. Multi-turn conversation learning (Auto-retain across chats)
  2. Context-aware prompt injection on subsequent questions
  3. Multi-user memory isolation: User A (Backend) vs User B (DevOps)
  4. Dynamic user profile synthesis via Hindsight reflection
  5. UI drawer data integrity across identity switches
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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED_DIR = os.path.join(ROOT_DIR, "Shared")
SCRIPTS_DIR = os.path.join(SHARED_DIR, "scripts")
for p in [SHARED_DIR, SCRIPTS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import chat_server
from chat_server import ChatHandler, ThreadedHTTPServer


class E2EMockOllama(http.server.BaseHTTPRequestHandler):
    """Mock Ollama Engine recording received prompts and streaming responses."""
    history = []

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(content_len).decode("utf-8")) if content_len > 0 else {}
        self.history.append(data)

        if self.path == "/api/chat":
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()

            # Generate response based on whether memory was present in prompt
            sys_msg = next((m.get("content", "") for m in data.get("messages", []) if m.get("role") == "system"), "")
            if "[HINDSIGHT LONG-TERM MEMORY]" in sys_msg and "ECS" in sys_msg:
                reply_text = "I see your Django on AWS ECS setup. Let us adjust your task definition memory limit to 1024MB."
            elif "[HINDSIGHT LONG-TERM MEMORY]" in sys_msg and "Kubernetes" in sys_msg:
                reply_text = "I recall your Golang Helm chart architecture. Let us define resource requests and limits."
            else:
                reply_text = "Standard response: Please describe your infrastructure and error message."

            chunk = json.dumps({
                "model": data.get("model", "llama3"),
                "message": {"role": "assistant", "content": reply_text},
                "done": True
            }) + "\n"
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()
        else:
            self.send_response(404)
            self.end_headers()


class E2EMockHindsight(http.server.BaseHTTPRequestHandler):
    """In-memory Hindsight mock simulating realistic semantic store and isolation."""
    banks = {}

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(content_len).decode("utf-8")) if content_len > 0 else {}

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
            content = items[0].get("content", "") if items and isinstance(items, list) else data.get("content", "")
            if bank not in self.banks:
                self.banks[bank] = []
            self.banks[bank].append({"content": content, "created_at": "2026-10-26T14:00:00Z"})

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "status": "retained"}).encode("utf-8"))

        elif self.path.endswith("/memories/recall") or self.path == "/v1/recall":
            query = data.get("query", "").lower()
            stored = self.banks.get(bank, [])

            matched = []
            for item in stored:
                txt = item["content"].lower()
                if any(w in txt for w in ["django", "ecs", "deploy", "kubernetes", "helm", "golang", "memory", "error"]):
                    matched.append(item["content"])

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "results": [{"text": m} for m in matched],
                "memories": matched
            }).encode("utf-8"))

        elif self.path.endswith("/reflect") or self.path == "/v1/reflect":
            stored = self.banks.get(bank, [])
            all_text = " ".join(m["content"] for m in stored)

            if "Django" in all_text and "ECS" in all_text:
                summary = "Senior Backend Lead specializing in Django and AWS ECS container deployments."
            elif "Kubernetes" in all_text and "Golang" in all_text:
                summary = "DevOps Lead managing Golang microservices and Helm charts on Kubernetes."
            else:
                summary = f"General developer profile with {len(stored)} recorded interactions."

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "text": summary,
                "response": summary
            }).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if "/memories" in self.path:
            bank = "default"
            if "/banks/" in self.path:
                parts = self.path.split("/")
                b_idx = parts.index("banks")
                if len(parts) > b_idx + 1:
                    bank = parts[b_idx + 1]
            stored = self.banks.get(bank, [])
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"items": [{"text": m["content"], "date": m["created_at"]} for m in stored], "memories": stored}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


class TestPhase4E2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. Start Mock Ollama
        cls.ollama_server = http.server.HTTPServer(("127.0.0.1", 0), E2EMockOllama)
        cls.ollama_port = cls.ollama_server.server_address[1]
        cls.ollama_thread = threading.Thread(target=cls.ollama_server.serve_forever, daemon=True)
        cls.ollama_thread.start()
        chat_server.OLLAMA_HOST = f"http://127.0.0.1:{cls.ollama_port}"

        # 2. Start Mock Hindsight
        cls.hindsight_server = http.server.HTTPServer(("127.0.0.1", 0), E2EMockHindsight)
        cls.hindsight_port = cls.hindsight_server.server_address[1]
        cls.hindsight_thread = threading.Thread(target=cls.hindsight_server.serve_forever, daemon=True)
        cls.hindsight_thread.start()

        # Connect chat_server cortex client
        if chat_server.cortex_client:
            chat_server.cortex_client.enabled = True
            chat_server.cortex_client.base_url = f"http://127.0.0.1:{cls.hindsight_port}"
            chat_server.cortex_client.user_id = "default"
            chat_server.cortex_client.auto_retain = True

        # 3. Start PortLLM Chat Server
        cls.chat_server = ThreadedHTTPServer(("127.0.0.1", 0), ChatHandler)
        cls.chat_port = cls.chat_server.server_address[1]
        cls.chat_url = f"http://127.0.0.1:{cls.chat_port}"
        cls.chat_thread = threading.Thread(target=cls.chat_server.serve_forever, daemon=True)
        cls.chat_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.chat_server.shutdown()
        cls.chat_server.server_close()
        cls.hindsight_server.shutdown()
        cls.hindsight_server.server_close()
        cls.ollama_server.shutdown()
        cls.ollama_server.server_close()

    def setUp(self):
        E2EMockOllama.history.clear()
        E2EMockHindsight.banks.clear()

    def _send_chat(self, prompt):
        payload = json.dumps({
            "model": "llama3:latest",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.chat_url}/ollama/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            memories_used = int(resp.headers.get("X-Cortex-Memories-Used", "0"))
            body = resp.read().decode("utf-8")
        return memories_used, body

    def _switch_user(self, user_id):
        payload = json.dumps({"user_id": user_id}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.chat_url}/api/cortex/user",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get_memories(self):
        req = urllib.request.Request(f"{self.chat_url}/api/cortex/memories")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get_profile(self):
        req = urllib.request.Request(f"{self.chat_url}/api/cortex/profile")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_01_full_e2e_multi_turn_learning_and_recall(self):
        """
        Verify multi-turn flow:
          Turn 1: User mentions tech stack -> 0 memories recalled -> background retained
          Turn 2: User asks follow-up -> memory recalled & injected -> contextual response
        """
        self._switch_user("rahul_backend")

        # Turn 1: Introduce stack
        memories_used_1, reply_1 = self._send_chat(
            "I'm configuring our Django application deployment on AWS ECS."
        )
        self.assertEqual(memories_used_1, 0)
        self.assertIn("Standard response", reply_1)

        # Allow background retain thread to persist
        time.sleep(0.3)

        # Verify bank now has memory
        bank_data = self._get_memories()
        self.assertEqual(len(bank_data.get("memories", [])), 1)

        # Turn 2: Follow-up regarding deployment error
        memories_used_2, reply_2 = self._send_chat(
            "Our deployment is failing with an out-of-memory error."
        )
        self.assertGreaterEqual(memories_used_2, 1)
        self.assertIn("Django on AWS ECS", reply_2)
        self.assertIn("1024MB", reply_2)

        # Verify upstream Ollama received memory block
        latest_req = E2EMockOllama.history[-1]
        sys_msg = next(m.get("content", "") for m in latest_req["messages"] if m.get("role") == "system")
        self.assertIn("[HINDSIGHT LONG-TERM MEMORY]", sys_msg)
        self.assertIn("Django", sys_msg)
        self.assertIn("ECS", sys_msg)

    def test_02_e2e_multi_user_memory_isolation(self):
        """
        Verify strict isolation between users:
          - Rahul's memories never leak to Priya
          - Priya's memories never leak to Rahul
        """
        # User 1: Rahul (Backend)
        self._switch_user("rahul")
        self._send_chat("I build Django REST APIs on AWS ECS.")
        time.sleep(0.3)

        # User 2: Priya (DevOps)
        self._switch_user("priya")
        self._send_chat("I maintain Kubernetes Helm charts for Golang services.")
        time.sleep(0.3)

        # Priya asks about deployment: should NOT receive Rahul's Django context
        priya_mems_used, priya_reply = self._send_chat("How to optimize deploy memory for services?")
        self.assertIn("Golang Helm chart", priya_reply)
        self.assertNotIn("Django", priya_reply)
        self.assertNotIn("ECS", priya_reply)

        # Verify Priya's memory inspector feed has only her items
        priya_feed = self._get_memories()
        self.assertEqual(priya_feed.get("bank_id"), "portllm-priya")
        for item in priya_feed.get("memories", []):
            self.assertNotIn("Django", item["content"])

        # Switch back to Rahul: verify his memories remain intact
        self._switch_user("rahul")
        rahul_feed = self._get_memories()
        self.assertEqual(rahul_feed.get("bank_id"), "portllm-rahul")
        rahul_contents = [m["content"] for m in rahul_feed.get("memories", [])]
        self.assertTrue(any("Django" in c for c in rahul_contents))
        self.assertFalse(any("Helm" in c for c in rahul_contents))

    def test_03_e2e_profile_reflection_synthesis(self):
        """Verify Hindsight reflect() builds accurate user persona summaries."""
        # Rahul profile check
        self._switch_user("rahul")
        self._send_chat("We use Django and AWS ECS for production.")
        time.sleep(0.3)
        prof_rahul = self._get_profile()
        self.assertIn("Senior Backend Lead", prof_rahul.get("profile", ""))
        self.assertIn("Django", prof_rahul.get("profile", ""))

        # Priya profile check
        self._switch_user("priya")
        self._send_chat("We configure Kubernetes clusters with Golang.")
        time.sleep(0.3)
        prof_priya = self._get_profile()
        self.assertIn("DevOps Lead", prof_priya.get("profile", ""))
        self.assertIn("Kubernetes", prof_priya.get("profile", ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
