"""
Chat export utility for LocalGPT Pro.
Exports conversation history to:
1. Clean Markdown (.md)
2. Professional PDF (.pdf) via ReportLab
Saved to data/users/<uid>/exports/ and provided for direct download.
"""
import os
from datetime import datetime
from typing import List, Dict, Any
from vector_store import get_user_storage_paths

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def export_chat_to_markdown(
    conversation: Dict[str, Any],
    messages: List[Dict[str, Any]],
    uid: str
) -> str:
    """
    Exports a conversation to a Markdown file.
    Returns the file path.
    """
    paths = get_user_storage_paths(uid)
    title = conversation.get("title", "Chat").replace(" ", "_")
    safe_title = "".join(c for c in title if c.isalnum() or c in ("-", "_"))
    filename = f"{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    file_path = os.path.join(paths["exports"], filename)

    lines = [
        f"# {conversation.get('title', 'SURESH.GPT Conversation')}",
        f"*Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC*",
        f"*Conversation ID: `{conversation.get('id', '')}`*",
        "\n---\n"
    ]

    for m in messages:
        role = m.get("role", "user").capitalize()
        content = m.get("content", "")
        timestamp = m.get("timestamp", "")
        sources = m.get("sources", [])

        lines.append(f"### {role} ({timestamp})")
        lines.append(content)
        lines.append("")

        if sources:
            lines.append("**Referenced Document Sources:**")
            for s in sources:
                lines.append(f"- 📄 `{s.get('source')}` (Page {s.get('page')})")
            lines.append("")

        lines.append("---\n")

    md_content = "\n".join(lines)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return file_path


def export_chat_to_pdf(
    conversation: Dict[str, Any],
    messages: List[Dict[str, Any]],
    uid: str
) -> str:
    """
    Exports a conversation to a formatted PDF using ReportLab.
    Returns the file path.
    """
    paths = get_user_storage_paths(uid)
    title = conversation.get("title", "Chat").replace(" ", "_")
    safe_title = "".join(c for c in title if c.isalnum() or c in ("-", "_"))
    filename = f"{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    file_path = os.path.join(paths["exports"], filename)

    doc = SimpleDocTemplate(
        file_path,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0284c7")
    )
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748b")
    )
    user_header_style = ParagraphStyle(
        'UserHeader',
        parent=styles['Heading3'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2563eb")
    )
    asst_header_style = ParagraphStyle(
        'AsstHeader',
        parent=styles['Heading3'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#059669")
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1e293b")
    )
    sources_style = ParagraphStyle(
        'SourcesStyle',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#475569")
    )

    story = []
    
    # Title & Metadata
    conv_title = conversation.get("title", "Conversation Export")
    story.append(Paragraph(f"SURESH.GPT — {conv_title}", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Model: Local Qwen2.5", meta_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=14))

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "").replace("\n", "<br/>")
        timestamp = m.get("timestamp", "")[:19].replace("T", " ")
        sources = m.get("sources", [])

        if role == "user":
            story.append(Paragraph(f"👤 User &bull; <font size=8 color='#64748b'>{timestamp}</font>", user_header_style))
        else:
            story.append(Paragraph(f"🤖 SURESH.GPT &bull; <font size=8 color='#64748b'>{timestamp}</font>", asst_header_style))
        
        story.append(Spacer(1, 4))
        story.append(Paragraph(content, body_style))
        story.append(Spacer(1, 6))

        if sources:
            source_texts = []
            for s in sources:
                source_texts.append(f"&bull; 📄 <b>{s.get('source')}</b> (p. {s.get('page')})")
            sources_html = "<b>Reference Sources:</b><br/>" + "<br/>".join(source_texts)
            story.append(Paragraph(sources_html, sources_style))
            story.append(Spacer(1, 6))

        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceAfter=10, spaceBefore=4))

    doc.build(story)
    return file_path

