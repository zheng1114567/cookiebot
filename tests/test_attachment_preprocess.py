from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pytest

from nanobot.agent import attachments
from nanobot.agent.attachments import preprocess_inbound_attachments
from nanobot.providers.base import LLMProvider, LLMResponse


class DummyProvider(LLMProvider):
    async def chat(self, *args, **kwargs):
        return LLMResponse(content="ok")

    def get_default_model(self) -> str:
        return "text-model"


def _write_docx(path: Path, text: str) -> None:
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>"
        + text
        + "</w:t></w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)


@pytest.mark.asyncio
async def test_pdf_is_converted_with_mineru_and_injected(monkeypatch, tmp_path: Path) -> None:
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(attachments.shutil, "which", lambda name: f"C:/bin/{name}.exe" if name == "mineru" else None)

    def fake_run(command, **kwargs):
        output_dir = Path(command[command.index("-o") + 1])
        converted = output_dir / "report" / "auto" / "report.md"
        converted.parent.mkdir(parents=True, exist_ok=True)
        converted.write_text("# Report\nMinerU extracted text.", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(attachments.subprocess, "run", fake_run)

    result = await preprocess_inbound_attachments([str(pdf)], supports_multimodal=False)

    assert result.unsupported_message is None
    assert result.media == []
    assert "## Uploaded Document Text" in result.content_suffix
    assert "MinerU extracted text." in result.content_suffix


@pytest.mark.asyncio
async def test_docx_is_extracted_and_injected(tmp_path: Path) -> None:
    docx = tmp_path / "brief.docx"
    _write_docx(docx, "Word document text")

    result = await preprocess_inbound_attachments([str(docx)], supports_multimodal=False)

    assert result.unsupported_message is None
    assert result.media == []
    assert "brief.docx" in result.content_suffix
    assert "Word document text" in result.content_suffix


@pytest.mark.asyncio
async def test_image_requires_multimodal_model(tmp_path: Path) -> None:
    image = tmp_path / "photo.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    result = await preprocess_inbound_attachments([str(image)], supports_multimodal=False)

    assert result.unsupported_message
    assert "does not support image input" in result.unsupported_message
    assert result.media == []


@pytest.mark.asyncio
async def test_image_is_kept_for_multimodal_model(tmp_path: Path) -> None:
    image = tmp_path / "photo.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    result = await preprocess_inbound_attachments([str(image)], supports_multimodal=True)

    assert result.unsupported_message is None
    assert result.media == [str(image)]


def test_default_multimodal_heuristic_is_conservative() -> None:
    provider = DummyProvider()

    assert provider.supports_multimodal("gpt-4o") is True
    assert provider.supports_multimodal("qwen2-vl-72b") is True
    assert provider.supports_multimodal("deepseek-chat") is False
