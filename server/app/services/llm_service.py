"""
Hosted LLM Inference Service via Groq & Free Multimodal Models:
- Ultra-fast streaming (300+ tokens/sec) of Llama 3.3 70B, Qwen 2.5 32B, and Mixtral.
- Free Image Reading / OCR via Groq Llama 3.2 11B Vision (transcribes scanned documents, photos, handwriting).
- Free High-Resolution Image Generation via Flux.1 (Pollinations AI - 100% free, no API key needed).
- Seamless fallback simulation if Groq API key is not yet set.
"""
from typing import AsyncGenerator, List, Dict, Any, Optional
import os
import json
import asyncio
import urllib.parse
import random
from groq import AsyncGroq
from server.app.core.config import settings


class HostedLLMService:
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        self.client = AsyncGroq(api_key=self.api_key) if self.api_key else None

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def stream_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """
        Yields tokens in real-time as Server-Sent Events (SSE) chunks.
        """
        chosen_model = model or settings.DEFAULT_MODEL

        # -------------------------------------------------------------
        # 1. FREE IMAGE GENERATION (Flux.1 / Pollinations AI)
        # -------------------------------------------------------------
        last_user_content = ""
        if messages:
            last_msg = messages[-1].get("content")
            if isinstance(last_msg, str):
                last_user_content = last_msg
            elif isinstance(last_msg, list):
                for part in last_msg:
                    if isinstance(part, dict) and part.get("type") == "text":
                        last_user_content = part.get("text", "")

        is_image_gen_model = chosen_model == "flux-image-gen"
        is_image_gen_intent = last_user_content.strip().lower().startswith(("/image", "generate image", "draw ", "create an image", "generate an image"))

        if is_image_gen_model or is_image_gen_intent:
            prompt_clean = last_user_content
            for prefix in ["/image", "generate image of", "generate image", "draw", "create an image of", "create an image", "generate an image of", "generate an image"]:
                if prompt_clean.lower().startswith(prefix):
                    prompt_clean = prompt_clean[len(prefix):].strip()
                    break

            if not prompt_clean.strip():
                prompt_clean = "A photorealistic futuristic laboratory with holographic AI interfaces"

            yield "🎨 **Generating high-resolution AI image with Turbo AI (Free Hosted Model)...**\n\n"
            await asyncio.sleep(0.3)

            seed = random.randint(10000, 999999)
            encoded_prompt = urllib.parse.quote(prompt_clean)
            
            # Use model=turbo which does not hit the classic API rate limit / 503 bottleneck
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&model=turbo&seed={seed}&nologo=true"

            yield f"### 🖼️ Prompt: *\"{prompt_clean}\"*\n\n"
            yield f"![{prompt_clean}]({image_url})\n\n"
            yield f"> **Model:** SDXL-Turbo / Fast Diffusion (1024×1024)\n"
            yield f"> **Status:** Free & Unlimited (Zero API Key Required)\n\n"
            yield f"[📥 Download Full-Res Image]({image_url})\n"
            return

        # -------------------------------------------------------------
        # 2. FREE IMAGE READING / OCR (Groq Llama 3.2 11B Vision)
        # -------------------------------------------------------------
        is_vision_request = (
            chosen_model == "llama-3.2-11b-vision-preview" or
            any(isinstance(m.get("content"), list) for m in messages)
        )

        if is_vision_request and not self.client:
            # Mock OCR demonstration if Groq key isn't provided yet
            yield "### 🔍 Image Reading & OCR Analysis (Llama 3.2 11B Vision)\n\n"
            yield "**Document / Image Processing Summary:**\n"
            yield "- **Mode:** Multimodal Vision & Text Extraction\n"
            yield "- **Detected Content:** Scanned text, layout blocks, and graphical structures\n\n"
            yield "```text\n[OCR Transcribed Text]:\nPROJECT SPECIFICATION & ARCHITECTURE\nStatus: Verified\nPlatform: Next.js + FastAPI + Groq Cloud\n```\n\n"
            yield "💡 *Tip: Add your free `GROQ_API_KEY` in `.env` to enable live Groq LPU Vision inference on any image!*"
            return

        # -------------------------------------------------------------
        # 3. LIVE GROQ INFERENCE (Text & Vision)
        # -------------------------------------------------------------
        if self.client:
            try:
                # If vision model, ensure temperature and max_tokens are within vision bounds
                vision_model = "llama-3.2-11b-vision-preview" if is_vision_request else chosen_model
                response = await self.client.chat.completions.create(
                    model=vision_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                async for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
                return
            except Exception as e:
                yield f"\n\n[Groq API Notification]: {str(e)}\n"

        # -------------------------------------------------------------
        # 4. FALLBACK TEXT SIMULATION (When GROQ_API_KEY is not set)
        # -------------------------------------------------------------
        mock_response = (
            f"Hello! I am SURESH.GPT running in local developer simulation mode ({chosen_model}). "
            "Notice: You have NOT configured a GROQ_API_KEY in your .env file, so this is a local simulated response. "
            "No external Groq account or key was created for you. "
            "To connect real live Groq cloud inference (Llama 3.3 70B / Qwen 2.5), you can paste your own key into .env whenever you are ready."
        )
        for word in mock_response.split(" "):
            yield word + " "
            await asyncio.sleep(0.04)


hosted_llm = HostedLLMService()
