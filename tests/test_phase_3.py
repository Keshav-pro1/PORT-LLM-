"""
Test Suite for Phase 3: Frontend UI Enhancements (Identity & Memory Inspector)
=============================================================================
Tests:
  1. Topbar Cortex Identity Bar HTML markup & IDs
  2. Memory Inspector Drawer HTML markup & IDs
  3. Cortex CSS class declarations and responsive styling
  4. Cortex client JavaScript function definitions
  5. X-Cortex-Memories-Used header extraction & badge rendering logic
  6. HTTP server static delivery of updated UI
"""

import http.server
import os
import sys
import threading
import unittest
import urllib.request

# Add project root and Shared to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED_DIR = os.path.join(ROOT_DIR, "Shared")
if SHARED_DIR not in sys.path:
    sys.path.insert(0, SHARED_DIR)

import chat_server
from chat_server import ChatHandler, ThreadedHTTPServer


class TestPhase3UIIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Read the actual PortLLM_UI.html file
        html_path = os.path.join(SHARED_DIR, "PortLLM_UI.html")
        with open(html_path, "r", encoding="utf-8") as f:
            cls.html_content = f.read()

        # Start PortLLM chat server on an ephemeral port
        cls.server = ThreadedHTTPServer(("127.0.0.1", 0), ChatHandler)
        cls.port = cls.server.server_address[1]
        cls.url = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_01_identity_bar_elements(self):
        """Verify presence of Topbar User Identity components."""
        html = self.html_content
        self.assertIn('id="cortex-identity-bar"', html)
        self.assertIn('id="cortex-user-select"', html)
        self.assertIn('id="cortex-add-user-btn"', html)
        self.assertIn('id="cortex-memory-badge"', html)
        self.assertIn('id="cortex-memory-count"', html)
        self.assertIn('switchCortexUser(this.value)', html)
        self.assertIn('promptAddCortexUser()', html)
        self.assertIn('toggleMemoryDrawer()', html)

    def test_02_memory_drawer_elements(self):
        """Verify presence of slide-out Memory Inspector Drawer components."""
        html = self.html_content
        self.assertIn('id="cortex-memory-drawer"', html)
        self.assertIn('id="cortex-bank-tag"', html)
        self.assertIn('id="cortex-user-profile"', html)
        self.assertIn('id="cortex-manual-memory"', html)
        self.assertIn('id="cortex-memories-list"', html)
        self.assertIn('refreshCortexProfile()', html)
        self.assertIn('refreshCortexMemories()', html)
        self.assertIn('manualRetainCortex()', html)

    def test_03_css_styling_classes(self):
        """Verify modern CSS classes for Cortex components."""
        html = self.html_content
        self.assertIn('.cortex-bar', html)
        self.assertIn('.cortex-badge', html)
        self.assertIn('.cortex-drawer', html)
        self.assertIn('.cortex-card', html)
        self.assertIn('.cortex-profile-content', html)
        self.assertIn('.cortex-mem-tag', html)
        self.assertIn('@keyframes cortexSlideIn', html)

    def test_04_javascript_functions(self):
        """Verify all frontend Cortex JavaScript functions exist."""
        html = self.html_content
        self.assertIn('async function initCortex()', html)
        self.assertIn('async function switchCortexUser(userId)', html)
        self.assertIn('function promptAddCortexUser()', html)
        self.assertIn('async function refreshCortexMemories()', html)
        self.assertIn('async function refreshCortexProfile()', html)
        self.assertIn('function toggleMemoryDrawer()', html)
        self.assertIn('async function manualRetainCortex()', html)

    def test_05_chat_memory_citation_hook(self):
        """Verify message citation extracts X-Cortex-Memories-Used and renders badge."""
        html = self.html_content
        self.assertIn("res.headers.get('X-Cortex-Memories-Used')", html)
        self.assertIn("aiMsg.memoriesUsed", html)
        self.assertIn("cortex-mem-tag", html)
        self.assertIn("recalled", html)

    def test_06_server_serves_cortex_ui(self):
        """Verify GET / on the chat server delivers the full Cortex UI."""
        req = urllib.request.Request(f"{self.url}/")
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            served_html = resp.read().decode("utf-8")
            self.assertIn('id="cortex-identity-bar"', served_html)
            self.assertIn('id="cortex-memory-drawer"', served_html)
            self.assertIn('initCortex()', served_html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
