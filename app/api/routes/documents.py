from __future__ import annotations

import io

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_document_ingestion_service
from app.core.security import hash_token
from app.services.document_ingestion import DocumentIngestionService

router = APIRouter(tags=["documents"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


@router.post("/documents/upload", status_code=201, summary="Upload a tenant document")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("en"),
    tenant_api_key: str | None = Header(default=None, alias="X-Tenant-API-Key"),
    ingestion: DocumentIngestionService = Depends(get_document_ingestion_service),
):
    if getattr(request.app.state, "database_backend", "mysql") != "postgres":
        raise HTTPException(status_code=503, detail="Document upload requires the PostgreSQL backend")
    if not tenant_api_key:
        raise HTTPException(status_code=401, detail="X-Tenant-API-Key is required")

    tenant_id = request.app.state.tenant_repo.resolve_api_key(hash_token(tenant_api_key))
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Invalid tenant API key")

    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="A valid filename is required")

    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Only .txt and text-based .pdf files are supported")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB limit")

    if extension == ".txt":
        try:
            extracted_text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="Text files must use UTF-8 encoding") from exc
    else:
        from PyPDF2 import PdfReader
        from PyPDF2.errors import PdfReadError

        try:
            reader = PdfReader(io.BytesIO(data), strict=False)
            if reader.is_encrypted:
                raise HTTPException(status_code=400, detail="Encrypted PDFs are not supported")
            extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except (PdfReadError, ValueError, EOFError) as exc:
            raise HTTPException(status_code=400, detail="The uploaded PDF could not be read") from exc

    try:
        result = await run_in_threadpool(
            ingestion.ingest_text,
            tenant_id,
            filename,
            extracted_text,
            language,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "document_id": result.document_id,
        "tenant_id": result.tenant_id,
        "filename": result.filename,
        "chunks": result.chunk_count,
    }
