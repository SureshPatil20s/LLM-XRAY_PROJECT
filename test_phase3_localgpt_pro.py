"""
Integration and Acceptance Test Suite for LocalGPT Pro (Phase 3).
Tests:
1. User profile and settings management
2. Multi-user data isolation (Chats & Messages)
3. Multi-user FAISS index & Document isolation
4. Chat export to Markdown and PDF (ReportLab)
5. Pinned chats & date grouping
6. Regression check on existing Phase 1 & 2 functionality
"""
import os
import shutil
import tempfile
import unittest
import numpy as np

import database
import rag
import vector_store
from memory import ConversationMemory
from export import export_chat_to_markdown, export_chat_to_pdf
from ui.sidebar import group_conversations_by_date


class TestLocalGPTProPhase3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        database.init_db()
        cls.user_a_uid = "test_user_a_suresh"
        cls.user_b_uid = "test_user_b_priya"

    def test_01_user_profile_and_settings(self):
        """Verify user upsertion, retrieval, and independent hyperparameter settings."""
        prof_a = database.upsert_user(
            uid=self.user_a_uid,
            email="suresh@test.com",
            display_name="Suresh",
            provider="email"
        )
        self.assertEqual(prof_a["display_name"], "Suresh")

        prof_b = database.upsert_user(
            uid=self.user_b_uid,
            email="priya@test.com",
            display_name="Priya Sharma",
            provider="google"
        )
        self.assertEqual(prof_b["display_name"], "Priya Sharma")

        # Set distinct settings for User A vs User B
        database.update_user_settings(self.user_a_uid, temperature=0.2, top_k=20, top_p=0.8, max_tokens=100)
        database.update_user_settings(self.user_b_uid, temperature=0.9, top_k=80, top_p=0.95, max_tokens=500)

        settings_a = database.get_user_settings(self.user_a_uid)
        settings_b = database.get_user_settings(self.user_b_uid)

        self.assertEqual(settings_a["temperature"], 0.2)
        self.assertEqual(settings_b["temperature"], 0.9)
        print("PASSED: User profile and settings isolation verified.")

    def test_02_per_user_chat_isolation(self):
        """Verify that User A cannot see User B's chats or messages."""
        # Clean any prior test chats
        database.delete_all_user_chats(self.user_a_uid)
        database.delete_all_user_chats(self.user_b_uid)

        # User A creates a chat
        conv_a_id = database.create_conversation("Suresh Private Physics Chat", uid=self.user_a_uid)
        database.add_message(conv_a_id, role="user", content="What is quantum mechanics?")
        database.add_message(conv_a_id, role="assistant", content="Quantum mechanics is physics at atomic scales.")

        # User B creates a chat
        conv_b_id = database.create_conversation("Priya Private AI Chat", uid=self.user_b_uid)
        database.add_message(conv_b_id, role="user", content="Explain Transformer Attention.")
        database.add_message(conv_b_id, role="assistant", content="Transformers use Q, K, V dot product attention.")

        # Check listings
        chats_a = database.get_conversations(uid=self.user_a_uid)
        chats_b = database.get_conversations(uid=self.user_b_uid)

        titles_a = [c["title"] for c in chats_a]
        titles_b = [c["title"] for c in chats_b]

        self.assertIn("Suresh Private Physics Chat", titles_a)
        self.assertNotIn("Priya Private AI Chat", titles_a)

        self.assertIn("Priya Private AI Chat", titles_b)
        self.assertNotIn("Suresh Private Physics Chat", titles_b)

        # Cross-access check
        cross_check = database.get_conversation(conv_a_id, uid=self.user_b_uid)
        self.assertIsNone(cross_check, "User B was able to fetch User A's conversation!")

        print("PASSED: Multi-user conversation data isolation verified.")

    def test_03_per_user_document_and_faiss_isolation(self):
        """Verify that User A's indexed documents are isolated from User B's FAISS index."""
        sample_text_a = "Suresh confidential document: Project Apollo code is 98765."
        sample_text_b = "Priya confidential document: Project Artemis code is 12345."

        # Ingest for User A
        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(sample_text_a)
            temp_path_a = f.name

        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(sample_text_b)
            temp_path_b = f.name

        try:
            rag.ingest_document(temp_path_a, "apollo.txt", uid=self.user_a_uid)
            rag.ingest_document(temp_path_b, "artemis.txt", uid=self.user_b_uid)

            # Query User A index for Artemis code
            ctx_a, _ = rag.retrieve_context("What is Artemis code?", uid=self.user_a_uid)
            self.assertNotIn("12345", ctx_a, "User A retrieved User B's secret document chunk!")

            # Query User B index for Apollo code
            ctx_b, _ = rag.retrieve_context("What is Apollo code?", uid=self.user_b_uid)
            self.assertNotIn("98765", ctx_b, "User B retrieved User A's secret document chunk!")

            # Query User A for Apollo
            ctx_a_valid, sources_a = rag.retrieve_context("Project Apollo", uid=self.user_a_uid)
            self.assertIn("98765", ctx_a_valid)
            self.assertEqual(sources_a[0]["source"], "apollo.txt")

            print("PASSED: User FAISS index and document isolation verified.")
        finally:
            if os.path.exists(temp_path_a):
                os.remove(temp_path_a)
            if os.path.exists(temp_path_b):
                os.remove(temp_path_b)

    def test_04_export_to_markdown_and_pdf(self):
        """Verify export of chat to Markdown and PDF via ReportLab."""
        conv_id = database.create_conversation("Exportable Research Chat", uid=self.user_a_uid)
        database.add_message(conv_id, role="user", content="Summarize Newton's first law.")
        database.add_message(
            conv_id,
            role="assistant",
            content="An object at rest remains at rest unless acted upon by an external force.",
            sources=[{"source": "principia.pdf", "page": 1, "snippet": "Law of inertia..."}]
        )

        conv = database.get_conversation(conv_id, uid=self.user_a_uid)
        messages = database.get_messages(conv_id)

        md_path = export_chat_to_markdown(conv, messages, uid=self.user_a_uid)
        self.assertTrue(os.path.exists(md_path))
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Summarize Newton's first law", content)
        self.assertIn("principia.pdf", content)

        pdf_path = export_chat_to_pdf(conv, messages, uid=self.user_a_uid)
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 500)

        print("PASSED: Export to Markdown and PDF verified.")

    def test_05_pinned_chats_and_date_grouping(self):
        """Verify chat pinning and chronological date groupings."""
        conv_id = database.create_conversation("Important Thesis", uid=self.user_a_uid)
        
        # Test toggle pin
        pinned_state = database.toggle_pin_conversation(conv_id)
        self.assertTrue(pinned_state)

        chats = database.get_conversations(uid=self.user_a_uid)
        grouped = group_conversations_by_date(chats)

        self.assertTrue(any(c["id"] == conv_id for c in grouped["Pinned"]))

        # Toggle unpin
        unpinned = database.toggle_pin_conversation(conv_id)
        self.assertFalse(unpinned)

        chats_unpinned = database.get_conversations(uid=self.user_a_uid)
        grouped_unpinned = group_conversations_by_date(chats_unpinned)
        self.assertTrue(any(c["id"] == conv_id for c in grouped_unpinned["Today"]))

        print("PASSED: Chat pinning and date grouping verified.")


if __name__ == "__main__":
    unittest.main()

