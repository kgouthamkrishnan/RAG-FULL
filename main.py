import os
import csv
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document

from rag_backend import index_documents, query_rag

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".pptx", ".html", ".htm"}
MAX_FILE_SIZE_MB = 30

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG API starting up...")
    yield
    logger.info("RAG API shutting down.")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Chatbot API",
    description="Document-grounded QA using FAISS + Gemini",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    text: str

class QueryRequest(BaseModel):
    query: str
    history: list[ChatMessage] = []

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query must not be empty.")
        return v.strip()

class QueryResponse(BaseModel):
    response: str

class UploadResponse(BaseModel):
    message: str
    chunks_indexed: int

# ── Custom Loaders ────────────────────────────────────────────────────────────
def load_txt(path: str) -> list[Document]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [Document(page_content=f.read(), metadata={"source": path})]

def load_md(path: str) -> list[Document]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [Document(page_content=f.read(), metadata={"source": path})]

def load_html(path: str) -> list[Document]:
    from html.parser import HTMLParser
    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.text_parts = []
            self._skip = False
        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self._skip = True
        def handle_endtag(self, tag):
            if tag in ("script", "style"):
                self._skip = False
        def handle_data(self, data):
            if not self._skip and data.strip():
                self.text_parts.append(data.strip())
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    parser = TextExtractor()
    parser.feed(raw)
    text = "\n".join(parser.text_parts)
    return [Document(page_content=text, metadata={"source": path})]

def load_csv(path: str) -> list[Document]:
    docs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        # Group every 50 rows into one chunk
        chunk_size = 50
        for i in range(0, len(rows), chunk_size):
            batch = rows[i:i + chunk_size]
            text = "\n".join(
                ", ".join(f"{k}: {v}" for k, v in row.items())
                for row in batch
            )
            docs.append(Document(page_content=text, metadata={"source": path, "rows": f"{i}-{i+len(batch)}"}))
    return docs

def load_xlsx(path: str) -> list[Document]:
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    docs = []
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        rows_text = []
        headers = None
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 0:
                headers = [str(c) if c is not None else "" for c in row]
            else:
                if any(c is not None for c in row):
                    if headers:
                        line = ", ".join(f"{h}: {v}" for h, v in zip(headers, row) if v is not None)
                    else:
                        line = ", ".join(str(c) for c in row if c is not None)
                    rows_text.append(line)
            # chunk every 50 rows
            if len(rows_text) >= 50:
                docs.append(Document(page_content="\n".join(rows_text), metadata={"source": path, "sheet": sheet}))
                rows_text = []
        if rows_text:
            docs.append(Document(page_content="\n".join(rows_text), metadata={"source": path, "sheet": sheet}))
    return docs

def load_pptx(path: str) -> list[Document]:
    from pptx import Presentation
    prs = Presentation(path)
    docs = []
    for i, slide in enumerate(prs.slides):
        texts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                texts.append(shape.text.strip())
        if texts:
            docs.append(Document(
                page_content="\n".join(texts),
                metadata={"source": path, "slide": i + 1}
            ))
    return docs

def load_file(path: str, ext: str) -> list[Document]:
    loaders = {
        ".pdf":  lambda: PyPDFLoader(path).load(),
        ".docx": lambda: Docx2txtLoader(path).load(),
        ".txt":  lambda: load_txt(path),
        ".md":   lambda: load_md(path),
        ".html": lambda: load_html(path),
        ".htm":  lambda: load_html(path),
        ".csv":  lambda: load_csv(path),
        ".xlsx": lambda: load_xlsx(path),
        ".pptx": lambda: load_pptx(path),
    }
    return loaders[ext]()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok"}

@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB.",
        )

    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(content)

        docs = load_file(temp_path, ext)

        if not docs:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract text from the uploaded file.",
            )

        chunks_indexed = index_documents(docs)
        logger.info("Uploaded and indexed: %s (%d chunks)", file.filename, chunks_indexed)

        return UploadResponse(
            message=f"'{file.filename}' uploaded and indexed successfully.",
            chunks_indexed=chunks_indexed,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during upload: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process file: {str(e)}",
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/query", response_model=QueryResponse, tags=["QA"])
async def query(request: QueryRequest):
    try:
        history = [m.model_dump() for m in request.history]
        answer = query_rag(request.query, chat_history=history)
        return QueryResponse(response=answer)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Error during query: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        )