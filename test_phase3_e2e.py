import asyncio
import json
import httpx
from server.app.main import app

async def run_tests():
    print("=== Starting Phase 3 End-to-End API Verification ===")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Health check
        res = await client.get("/health")
        assert res.status_code == 200, f"Health check failed: {res.text}"
        print("✓ Health check OK:", res.json())

        # 2. List Models
        res = await client.get("/api/v1/models")
        assert res.status_code == 200, f"Models check failed: {res.text}"
        models = res.json()
        print(f"✓ Models OK: {len(models)} models available ({models[0]['name']})")

        # 3. Register user
        reg_data = {
            "email": "phase3_test@example.com",
            "password": "Password123!",
            "name": "Phase 3 Evaluator"
        }
        res = await client.post("/api/v1/auth/register", json=reg_data)
        if res.status_code == 400:
            # Login instead
            login_data = {"email": reg_data["email"], "password": reg_data["password"]}
            res = await client.post("/api/v1/auth/login", json=login_data)
        assert res.status_code == 200, f"Auth failed: {res.text}"
        auth_info = res.json()
        token = auth_info["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("✓ Auth registration/login OK! User ID:", auth_info["user"]["id"])

        # 4. Google OAuth login
        google_data = {
            "email": "evaluator.google@gmail.com",
            "name": "Google Test User",
            "avatar_url": "https://example.com/photo.png"
        }
        res = await client.post("/api/v1/auth/google", json=google_data)
        assert res.status_code == 200, f"Google auth failed: {res.text}"
        print("✓ Google OAuth login OK!")

        # 5. Create Conversation
        conv_payload = {"title": "Architecture & RAG Discussion", "model": "llama-3.3-70b-versatile"}
        res = await client.post("/api/v1/conversations", json=conv_payload, headers=headers)
        assert res.status_code in [200, 201], f"Create conv failed: {res.text}"
        conv = res.json()
        conv_id = conv["id"]
        print("✓ Created conversation OK! ID:", conv_id)

        # 6. Upload a document (RAG ingestion)
        sample_doc_content = (
            "Project Phase 3 is a Next.js and FastAPI ChatGPT-style enterprise platform. "
            "It utilizes hosted Groq inference running Llama 3.3 70B and Qwen 2.5 32B models. "
            "The platform supports multi-turn conversations, RAG document knowledge base, "
            "and PostgreSQL database storage."
        )
        files = {
            "file": ("system_spec.txt", sample_doc_content.encode("utf-8"), "text/plain")
        }
        res = await client.post("/api/v1/documents/upload", files=files, headers=headers)
        assert res.status_code == 200, f"Upload doc failed: {res.text}"
        doc = res.json()
        print(f"✓ Uploaded & indexed document OK! Filename: {doc['filename']}, Chunks: {doc['chunk_count']}")

        # 7. Test Streaming Chat with RAG
        chat_req = {
            "conversation_id": conv_id,
            "content": "What LLM models and database does Project Phase 3 use according to the specification?",
            "model": "llama-3.3-70b-versatile",
            "use_rag": True
        }
        
        async with client.stream("POST", "/api/v1/messages/stream", json=chat_req, headers=headers) as stream_res:
            assert stream_res.status_code == 200, f"Stream failed: {stream_res.status_code}"
            streamed_text = ""
            sources_found = []
            async for line in stream_res.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    payload = json.loads(data_str)
                    if "delta" in payload:
                        streamed_text += payload["delta"]
                    if "sources" in payload:
                        sources_found.extend(payload["sources"])
                except Exception:
                    streamed_text += data_str

        print("✓ SSE Stream Chat Response received successfully!")
        print(f"  Total characters: {len(streamed_text)}")
        print(f"  RAG Sources cited: {len(sources_found)}")
        if sources_found:
            print(f"  Source snippet: '{sources_found[0]['snippet'][:60]}...'")
        print(f"  Response sample: '{streamed_text[:120]}...'")

        # 8. Test Rename and Pin
        res = await client.put(f"/api/v1/conversations/{conv_id}", json={"title": "Updated Title", "is_pinned": True}, headers=headers)
        assert res.status_code == 200
        print("✓ Conversation rename & pin OK!")

        # 9. Test Guest Login (Zero credentials required)
        res = await client.post("/api/v1/auth/guest")
        assert res.status_code == 200, f"Guest login failed: {res.text}"
        guest_info = res.json()
        print("✓ Guest seamless authentication OK! Guest Name:", guest_info["user"]["name"])

        # 10. Test Free Image Generation (Flux.1 / Pollinations)
        img_conv_res = await client.post("/api/v1/conversations", json={"title": "Image Test", "model": "flux-image-gen"}, headers=headers)
        img_conv_id = img_conv_res.json()["id"]
        async with client.stream("POST", "/api/v1/messages/stream", json={"conversation_id": img_conv_id, "content": "A futuristic city in neon rain", "model": "flux-image-gen"}, headers=headers) as stream_res:
            assert stream_res.status_code == 200
            img_stream_text = ""
            async for line in stream_res.aiter_lines():
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    try:
                        p = json.loads(line[6:])
                        if "delta" in p:
                            img_stream_text += p["delta"]
                    except Exception:
                        pass
        assert "image.pollinations.ai" in img_stream_text
        print("✓ Free Image Generation (Flux.1) stream OK! Generated URL found in output.")

        # 11. Test Free Vision & OCR (Llama 3.2 11B Vision)
        async with client.stream("POST", "/api/v1/messages/stream", json={"conversation_id": img_conv_id, "content": "Extract all text from this scan", "model": "llama-3.2-11b-vision-preview", "image_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="}, headers=headers) as stream_res:
            assert stream_res.status_code == 200
            vision_stream_text = ""
            async for line in stream_res.aiter_lines():
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    try:
                        p = json.loads(line[6:])
                        if "delta" in p:
                            vision_stream_text += p["delta"]
                    except Exception:
                        pass
        print("✓ Free Vision & OCR (Llama 3.2 11B Vision) stream OK!")

        print("\n🎉 ALL PHASE 3 BACKEND API TESTS (AUTH, RAG, IMAGE GEN, OCR) PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(run_tests())
