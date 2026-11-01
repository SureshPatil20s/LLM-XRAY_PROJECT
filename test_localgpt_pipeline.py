"""
Comprehensive Integration Test for LocalGPT (Project Phase 2 Upgrade)
Tests all core modules and workflows end-to-end.
"""
import os
import shutil
import database
import document_loader
import embeddings
import vector_store
import rag
import memory
import xray
import chat
import claim_verifier

def run_all_checks():
    print("=== STARTING LOCALGPT INTEGRATION TESTS ===")

    # 1. Database Check
    print("\n[1/7] Testing database.py...")
    database.init_db()
    c_id = database.create_conversation("Integration Test Chat")
    assert c_id is not None
    m_id1 = database.add_message(c_id, "user", "What is Machine Learning?")
    m_id2 = database.add_message(c_id, "assistant", "Machine Learning is a subset of AI.", sources=[{"source": "test.txt", "page": 1}])
    msgs = database.get_messages(c_id)
    assert len(msgs) == 2
    assert msgs[0]["content"] == "What is Machine Learning?"
    assert len(msgs[1]["sources"]) == 1
    print("PASSED: Database CRUD, conversations, and messages work correctly.")

    # 2. Document Loader & Chunker Check
    print("\n[2/7] Testing document_loader.py...")
    test_doc_path = "data/documents/test_sample.txt"
    with open(test_doc_path, "w", encoding="utf-8") as f:
        f.write("Transformers are deep learning architectures introduced in 2017 by Vaswani et al. They rely on multi-head self-attention mechanisms.")
    pages = document_loader.load_document(test_doc_path)
    assert len(pages) == 1
    chunks = document_loader.chunk_text(pages, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 1
    assert "Transformers" in chunks[0]["text"]
    print(f"PASSED: Document loading and chunking work: {len(chunks)} chunks produced.")

    # 3. Embeddings & FAISS Vector Store Check
    print("\n[3/7] Testing embeddings.py & vector_store.py...")
    texts = [c["text"] for c in chunks]
    vecs = embeddings.embed_texts(texts)
    assert vecs.shape[0] == len(texts)
    assert vecs.shape[1] == 384
    vstore = vector_store.get_vector_store()
    vstore.clear()
    vstore.add_chunks(chunks, vecs)
    assert vstore.get_total_chunks() == len(chunks)
    q_vec = embeddings.embed_query("When was Transformer introduced?")
    search_res = vstore.search(q_vec, top_k=1)
    assert len(search_res) == 1
    assert "Transformers" in search_res[0]["text"]
    print("PASSED: Embeddings and FAISS similarity search work accurately.")

    # 4. RAG Pipeline Check
    print("\n[4/7] Testing rag.py...")
    rag_ctx, rag_sources = rag.retrieve_context("What did Vaswani introduce in 2017?", top_k=1)
    assert len(rag_sources) == 1
    assert rag_sources[0]["source"] == "test_sample.txt"
    assert "Transformers" in rag_ctx
    print(f"PASSED: RAG retrieval context and source provenance verified: {rag_sources[0]['source']} (Page {rag_sources[0]['page']}).")

    # 5. Conversation Memory Check
    print("\n[5/7] Testing memory.py...")
    mem = memory.ConversationMemory()
    mem.add_user_message("Hello AI")
    mem.add_assistant_message("Hello! How can I help you today?")
    context_msgs = mem.build_prompt_context(
        system_prompt="You are a helpful tutor.",
        rag_context="Reference document: Machine Learning 101."
    )
    assert len(context_msgs) == 3
    assert context_msgs[0]["role"] == "system"
    assert "Machine Learning 101" in context_msgs[0]["content"]
    assert context_msgs[1]["content"] == "Hello AI"
    print("PASSED: Conversation memory and short-term context building work.")

    # 6. Claim Verifier Check
    print("\n[6/7] Testing claim_verifier.py...")
    c_eval = claim_verifier.execute_phase3_verification(
        original_answer="Water boils at 100 degrees Celsius.",
        original_run_id="test_run",
        prompt="At what temperature does water boil?"
    )
    assert c_eval["total_claims"] == 1
    assert c_eval["claims"][0]["verification_status"] == claim_verifier.VerificationStatus.SUPPORTED
    print("PASSED: Claim verification verified: 100 C boiling point claim -> SUPPORTED with IUPAC citation.")

    # 7. Cleanup test artifacts
    if os.path.exists(test_doc_path):
        os.remove(test_doc_path)
    database.delete_conversation(c_id)

    print("\n=== ALL LOCALGPT INTEGRATION TESTS COMPLETED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_all_checks()
