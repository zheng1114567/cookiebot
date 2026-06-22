"""Inbound attachment preprocessing for documents and images."""

from __future__ import annotations

import asyncio
import mimetypes
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

from loguru import logger

from nanobot.utils.helpers import detect_image_mime


@dataclass
class AttachmentPreprocessResult:
    """Result of preparing inbound media for an LLM turn."""

    content_suffix: str = ""
    media: list[str] = field(default_factory=list)
    unsupported_message: str | None = None


def _is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def _is_docx(path: Path) -> bool:
    return path.suffix.lower() == ".docx"


def _is_image(path: Path) -> bool:
    try:
        raw = path.read_bytes()[:32]
    except OSError:
        raw = b""
    mime = detect_image_mime(raw) or mimetypes.guess_type(path.name)[0]
    return bool(mime and mime.startswith("image/"))


def _clip_document_text(text: str, limit: int = 40_000) -> str:
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\n[... document text truncated ...]"


def _read_docx_text(path: Path) -> str:
    """Extract plain paragraph text from a DOCX using only stdlib."""
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        parts = [
            node.text or ""
            for node in paragraph.findall(".//w:t", namespace)
            if node.text
        ]
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _collect_mineru_text(output_dir: Path) -> str:
    candidates = sorted(
        [
            *output_dir.rglob("*.md"),
            *output_dir.rglob("*.txt"),
        ],
        key=lambda p: (p.suffix != ".md", len(p.parts), str(p)),
    )
    parts = []
    for candidate in candidates:
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            continue
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _run_mineru_sync(path: Path) -> str:
    """Run MinerU/Magic-PDF and return extracted markdown/text."""
    commands = []
    mineru = shutil.which("mineru")
    magic_pdf = shutil.which("magic-pdf")
    with tempfile.TemporaryDirectory(prefix="nanobot_mineru_") as tmp:
        output_dir = Path(tmp)
        if mineru:
            commands.append([mineru, "-p", str(path), "-o", str(output_dir)])
        if magic_pdf:
            commands.append([magic_pdf, "-p", str(path), "-o", str(output_dir)])
        if not commands:
            raise RuntimeError("MinerU is not installed or not available on PATH.")

        errors: list[str] = []
        for command in commands:
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(path.parent),
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=False,
                )
            except Exception as exc:
                errors.append(str(exc))
                continue
            if completed.returncode != 0:
                errors.append((completed.stderr or completed.stdout or "").strip())
                continue
            text = _collect_mineru_text(output_dir)
            if text:
                return text
        detail = "; ".join(error for error in errors if error) or "no text output produced"
        raise RuntimeError(f"MinerU PDF conversion failed: {detail}")


async def _read_pdf_text_with_mineru(path: Path) -> str:
    return await asyncio.to_thread(_run_mineru_sync, path)


async def preprocess_inbound_attachments(
    media: list[str] | None,
    *,
    supports_multimodal: bool,
) -> AttachmentPreprocessResult:
    """Convert document attachments to text and keep supported image media."""
    if not media:
        return AttachmentPreprocessResult()

    result = AttachmentPreprocessResult()
    document_blocks: list[str] = []
    unsupported_images: list[str] = []
    failures: list[str] = []

    for raw_path in media:
        path = Path(raw_path)
        if not path.is_file():
            result.media.append(raw_path)
            continue
        if _is_image(path):
            if supports_multimodal:
                result.media.append(raw_path)
            else:
                unsupported_images.append(str(path))
            continue
        if _is_pdf(path):
            try:
                text = await _read_pdf_text_with_mineru(path)
                document_blocks.append(
                    f"### {path.name}\n{_clip_document_text(text)}"
                )
            except Exception as exc:
                logger.warning("Failed to convert PDF with MinerU: {} ({})", path, exc)
                failures.append(f"{path.name}: {exc}")
            continue
        if _is_docx(path):
            try:
                text = await asyncio.to_thread(_read_docx_text, path)
                document_blocks.append(
                    f"### {path.name}\n{_clip_document_text(text)}"
                )
            except Exception as exc:
                logger.warning("Failed to extract DOCX text: {} ({})", path, exc)
                failures.append(f"{path.name}: {exc}")
            continue
        result.media.append(raw_path)

    if unsupported_images:
        names = ", ".join(Path(p).name for p in unsupported_images)
        result.unsupported_message = (
            "The current model does not support image input. "
            f"Please switch to a multimodal model to analyze: {names}"
        )
        return result

    if document_blocks:
        result.content_suffix = (
            "\n\n## Uploaded Document Text\n"
            + "\n\n".join(document_blocks)
        )
    if failures and not document_blocks:
        result.unsupported_message = (
            "I could not extract text from the uploaded document(s): "
            + "; ".join(failures)
        )
    elif failures:
        result.content_suffix += (
            "\n\n## Attachment Extraction Warnings\n"
            + "\n".join(f"- {failure}" for failure in failures)
        )
    return result
