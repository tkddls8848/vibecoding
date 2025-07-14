#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
server.py: Claude RAG 시스템 서버 및 웹 인터페이스

이 모듈은 다음 기능들을 통합합니다:
- FastAPI 서버 구현 (main.py)
- 웹 인터페이스 제공
- 임베딩 생성 및 관리
- 서버 실행 스크립트 (run_rag.py)
"""

import os
import logging
import argparse
import sys
import subprocess
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

# claude_rag.py 모듈에서 필요한 함수와 클래스 가져오기
from claude_rag import (
    CONFIG, DocumentEmbedder, query_claude_rag
)

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("claude_rag.server")

# 현재 파일 경로 기준으로 템플릿 및 정적 파일 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(BASE_DIR, "templates")
static_dir = os.path.join(BASE_DIR, "static")

# 템플릿 디렉토리가 없으면 생성
os.makedirs(templates_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)

# 모델 정의
class RagQueryRequest(BaseModel):
    q: str = Field(..., description="검색어")
    top_k: int = Field(CONFIG["top_k"], description="검색할 상위 문서 수")
    temperature: float = Field(CONFIG["temperature"], description="응답의 창의성")
    max_tokens: int = Field(CONFIG["max_tokens"], description="최대 토큰 수")

class RagQueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    status: str

# 임베딩 생성 함수
def create_embeddings(data_dir, embed_dir=None, chunk_size=None, overlap=None):
    """임베딩 생성 함수"""
    logger.info(f"'{data_dir}' 디렉토리의 문서를 임베딩합니다...")
    
    # DocumentEmbedder 클래스 사용
    embedder = DocumentEmbedder()
    embedder.create_embeddings(
        data_dir=data_dir,
        output_dir=embed_dir or CONFIG["embed_dir"],
        chunk_size=chunk_size or CONFIG["chunk_size"],
        overlap=overlap or CONFIG["chunk_overlap"]
    )
    
    logger.info(f"✅ 임베딩이 '{embed_dir or CONFIG['embed_dir']}' 디렉토리에 저장되었습니다.")

# 임베딩 확인 및 생성 함수
def check_and_create_embeddings(data_dir, embed_dir, chunk_size=None, overlap=None, force_embedding=False):
    """임베딩 파일이 존재하는지 확인하고, 없으면 생성"""
    # 임베딩 파일 경로
    index_path = os.path.join(embed_dir, "faiss_index.bin")
    docs_path = os.path.join(embed_dir, "chunked_docs.pkl")
    
    # 임베딩 디렉토리 생성
    os.makedirs(embed_dir, exist_ok=True)
    
    # 임베딩 파일 존재 확인
    if force_embedding or not os.path.exists(index_path) or not os.path.exists(docs_path):
        logger.info("임베딩 파일이 존재하지 않거나 강제 재생성 옵션이 설정되었습니다.")
        
        # 데이터 디렉토리 확인
        if not os.path.exists(data_dir):
            logger.error(f"❌ 데이터 디렉토리 '{data_dir}'가 존재하지 않습니다.")
            logger.error(f"데이터 디렉토리를 생성하고 텍스트 파일을 추가한 후 다시 시도하세요.")
            return False
        
        # 지원하는 파일 확인
        supported_extensions = CONFIG["supported_extensions"]
        all_files = []
        for ext in supported_extensions:
            ext = ext.strip()
            if not ext.startswith('.'):
                ext = '.' + ext
            files = list(Path(data_dir).glob(f"*{ext}"))
            all_files.extend(files)
        
        if not all_files:
            logger.error(f"❌ 데이터 디렉토리 '{data_dir}'에 지원하는 파일({supported_extensions})이 없습니다.")
            logger.error(f"지원하는 형식의 파일을 추가한 후 다시 시도하세요.")
            return False
        
        logger.info(f"📄 데이터 디렉토리 '{data_dir}'에서 {len(all_files)}개의 파일을 찾았습니다.")
        logger.info(f"지원하는 파일 형식: {supported_extensions}")
        
        # 임베딩 생성
        create_embeddings(
            data_dir=data_dir,
            embed_dir=embed_dir,
            chunk_size=chunk_size,
            overlap=overlap
        )
        
        return True
    else:
        logger.info(f"✅ 임베딩 파일이 이미 존재합니다: {index_path}")
        return True

# 기본 HTML 템플릿 생성
def create_templates():
    # index.html 파일이 없으면 생성
    index_html_path = os.path.join(templates_dir, "index.html")
    if not os.path.exists(index_html_path):
        with open(index_html_path, "w", encoding="utf-8") as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Claude RAG 쿼리 인터페이스</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        form {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
        }
        input[type="text"], textarea {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
            font-size: 16px;
        }
        textarea {
            min-height: 100px;
            resize: vertical;
        }
        input[type="number"] {
            width: 100px;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 15px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
        }
        .result {
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            margin-top: 20px;
            white-space: pre-wrap;
        }
        .documents {
            margin-top: 20px;
            border-top: 1px solid #ddd;
            padding-top: 10px;
        }
        .document {
            background-color: #f0f0f0;
            border-radius: 4px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .document-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .params-group {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 15px;
        }
        .param {
            flex: 1;
            min-width: 150px;
        }
        .file-type {
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Claude RAG 쿼리 인터페이스</h1>
        <p>아래 입력 필드에 질문을 입력하고 제출 버튼을 클릭하세요.</p>
        <p><strong>지원하는 파일 형식:</strong> .md (마크다운), .txt (텍스트)</p>
        
        <form method="get" action="/web-query">
            <div>
                <label for="q">질문:</label>
                <textarea name="q" id="q" required>{{ query }}</textarea>
            </div>
            
            <div class="params-group">
                <div class="param">
                    <label for="top_k">검색 문서 수:</label>
                    <input type="number" name="top_k" id="top_k" value="{{ top_k }}" min="1" max="20">
                </div>
                <div class="param">
                    <label for="temperature">응답 창의성:</label>
                    <input type="number" name="temperature" id="temperature" value="{{ temperature }}" min="0" max="1" step="0.1">
                </div>
                <div class="param">
                    <label for="max_tokens">최대 토큰 수:</label>
                    <input type="number" name="max_tokens" id="max_tokens" value="{{ max_tokens }}" min="100" max="4000">
                </div>
            </div>
            
            <button type="submit">질문 제출</button>
        </form>
        
        {% if result %}
        <h2>응답:</h2>
        <div class="result">{{ result }}</div>
        
        {% if used_documents %}
        <div class="documents">
            <h3>사용된 문서:</h3>
            {% for doc in used_documents %}
            <div class="document">
                <div class="document-title">
                    {{ doc.file_name }} 
                    <span class="file-type">({{ doc.file_extension }} 파일, 유사도: {{ "%.4f"|format(doc.score) }})</span>
                </div>
                <div class="document-preview">{{ doc.preview }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        {% endif %}
    </div>
</body>
</html>""")
        logger.info(f"HTML 템플릿 생성 완료: {index_html_path}")

# Lifespan 이벤트 핸들러
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행되는 코드
    logger.info("Claude RAG FastAPI 서버 시작 중...")
    logger.info(f"임베딩 디렉토리: {CONFIG['embed_dir']}")
    logger.info(f"인덱스 파일: {CONFIG['index_path']}")
    logger.info(f"문서 파일: {CONFIG['docs_path']}")
    logger.info(f"지원하는 파일 형식: {CONFIG['supported_extensions']}")
    
    # HTML 템플릿 생성
    create_templates()
    
    # API 키 존재 확인 및 상태 출력
    if CONFIG["anthropic_api_key"]:
        logger.info("✅ Anthropic API Key가 설정되었습니다.")
    else:
        logger.warning("❌ Anthropic API Key가 설정되지 않았습니다. API 호출이 실패할 것입니다.")
    
    if CONFIG["openapi_key"]:
        logger.info("✅ OpenAPI Key(serviceKey)가 설정되었습니다.")
    else:
        logger.warning("❌ OpenAPI Key(serviceKey)가 설정되지 않았습니다. API URL 생성 시 인증 파라미터가 누락될 수 있습니다.")
    
    # 필요한 파일 존재 확인
    if not os.path.exists(CONFIG["index_path"]):
        logger.warning(f"인덱스 파일({CONFIG['index_path']})이 존재하지 않습니다. 서버는 실행되지만 쿼리가 실패할 수 있습니다.")
    
    if not os.path.exists(CONFIG["docs_path"]):
        logger.warning(f"문서 파일({CONFIG['docs_path']})이 존재하지 않습니다. 서버는 실행되지만 쿼리가 실패할 수 있습니다.")
    
    yield  # 애플리케이션 실행
    
    # 종료 시 실행되는 코드
    logger.info("Claude RAG FastAPI 서버 종료 중...")

# FastAPI 앱 생성
app = FastAPI(
    title="Claude RAG API",
    description="조달청 나라장터 API 문서 기반의 Claude RAG API (마크다운 및 텍스트 파일 지원)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 템플릿 엔진 설정
templates = Jinja2Templates(directory=templates_dir)

# 정적 파일 설정
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """웹 인터페이스 메인 페이지"""
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "query": "",
            "top_k": CONFIG["top_k"],
            "temperature": CONFIG["temperature"],
            "max_tokens": CONFIG["max_tokens"]
        }
    )

@app.get("/web-query", response_class=HTMLResponse)
async def web_query(
    request: Request,
    q: str,
    top_k: int = CONFIG["top_k"],
    temperature: float = CONFIG["temperature"],
    max_tokens: int = CONFIG["max_tokens"]
):
    """웹 인터페이스용 RAG 쿼리 처리"""
    try:
        # API 키 확인
        if not CONFIG["anthropic_api_key"]:
            return templates.TemplateResponse(
                "index.html", 
                {
                    "request": request,
                    "query": q,
                    "top_k": top_k,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "result": "오류: ANTHROPIC_API_KEY가 설정되지 않았습니다. .env 파일에 API 키를 설정해주세요."
                }
            )
        
        # 파일 존재 확인
        if not os.path.exists(CONFIG["index_path"]) or not os.path.exists(CONFIG["docs_path"]):
            return templates.TemplateResponse(
                "index.html", 
                {
                    "request": request,
                    "query": q,
                    "top_k": top_k,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "result": "오류: 임베딩 파일이 존재하지 않습니다. 먼저 임베딩을 생성하세요."
                }
            )
        
        # Claude RAG API 호출
        result = await query_claude_rag(
            query=q,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k
        )
        
        # 결과 추출
        if result["status"] == "success" and result["results"]:
            response_content = result["results"][0]["content"]
            used_documents = result["results"][0]["used_documents"] if "used_documents" in result["results"][0] else []
        else:
            response_content = "오류: 응답을 받지 못했습니다."
            used_documents = []
        
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "query": q,
                "top_k": top_k,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "result": response_content,
                "used_documents": used_documents
            }
        )
        
    except Exception as e:
        logger.error(f"웹 쿼리 처리 오류: {str(e)}")
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "query": q,
                "top_k": top_k,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "result": f"오류: 쿼리 처리 중 오류가 발생했습니다: {str(e)}"
            }
        )

@app.get("/query")
async def get_query(q: str, top_k: int = CONFIG["top_k"], temperature: float = CONFIG["temperature"], max_tokens: int = CONFIG["max_tokens"]):
    """GET 요청을 통한 RAG 쿼리 엔드포인트"""
    try:
        # API 키 확인
        if not CONFIG["anthropic_api_key"]:
            raise HTTPException(
                status_code=500,
                detail="ANTHROPIC_API_KEY가 설정되지 않았습니다. .env 파일에 API 키를 설정해주세요."
            )
        
        # 파일 존재 확인
        if not os.path.exists(CONFIG["index_path"]) or not os.path.exists(CONFIG["docs_path"]):
            raise HTTPException(
                status_code=500,
                detail="임베딩 파일이 존재하지 않습니다. 먼저 임베딩을 생성하세요."
            )
        
        result = await query_claude_rag(
            query=q,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k
        )
        
        return result
        
    except Exception as e:
        logger.error(f"쿼리 처리 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"쿼리 처리 중 오류가 발생했습니다: {str(e)}"
        )

@app.post("/query", response_model=RagQueryResponse)
async def rag_query(request: RagQueryRequest):
    """RAG 쿼리 엔드포인트"""
    try:
        # API 키 확인
        if not CONFIG["anthropic_api_key"]:
            raise HTTPException(
                status_code=500,
                detail="ANTHROPIC_API_KEY가 설정되지 않았습니다. .env 파일에 API 키를 설정해주세요."
            )
        
        # 파일 존재 확인
        if not os.path.exists(CONFIG["index_path"]) or not os.path.exists(CONFIG["docs_path"]):
            raise HTTPException(
                status_code=500,
                detail="임베딩 파일이 존재하지 않습니다. 먼저 임베딩을 생성하세요."
            )
        
        result = await query_claude_rag(
            query=request.q,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_k=request.top_k
        )
        
        return result
        
    except Exception as e:
        logger.error(f"쿼리 처리 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"쿼리 처리 중 오류가 발생했습니다: {str(e)}"
        )

# 통합 실행 함수 (원래 run_rag.py)
def run_server(host="0.0.0.0", port=8001, data_dir=None, embed_dir=None, chunk_size=None, overlap=None, force_embedding=False):
    """서버 실행 함수"""
    # 기본값 설정
    embed_dir = embed_dir or CONFIG["embed_dir"]
    data_dir = data_dir or "../data"
    
    # 임베딩 디렉토리 경로
    embed_dir = os.path.abspath(embed_dir)
    data_dir = os.path.abspath(data_dir)
    
    # 임베딩 확인 및 생성
    embedding_ready = check_and_create_embeddings(
        data_dir=data_dir,
        embed_dir=embed_dir,
        chunk_size=chunk_size,
        overlap=overlap,
        force_embedding=force_embedding
    )
    
    if not embedding_ready:
        logger.error("임베딩 준비에 실패했습니다. 서버를 시작하지 않습니다.")
        sys.exit(1)
    
    logger.info(f"🚀 Claude RAG API 서버를 {host}:{port}에서 시작합니다...")
    
    # uvicorn 서버 실행
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claude RAG API 서버")
    parser.add_argument("--host", type=str, default="localhost", help="서버 호스트 주소")
    parser.add_argument("--port", type=int, default=8001, help="서버 포트 번호")
    parser.add_argument("--data-dir", type=str, default="./data", help="임베딩할 데이터 디렉토리 경로")
    parser.add_argument("--embed-dir", type=str, default=CONFIG["embed_dir"], help="임베딩 디렉토리 경로")
    parser.add_argument("--chunk-size", type=int, default=CONFIG["chunk_size"], help="문서 청크 크기")
    parser.add_argument("--overlap", type=int, default=CONFIG["chunk_overlap"], help="문서 청크 간 겹치는 영역 크기")
    parser.add_argument("--api-key", type=str, help="Anthropic API 키 (환경 변수보다 우선)")
    parser.add_argument("--force-embedding", action="store_true", help="임베딩을 강제로 재생성합니다")
    
    args = parser.parse_args()
    
    # 명령줄 인수로 API 키가 제공되면 환경 변수로 설정
    if args.api_key:
        os.environ["ANTHROPIC_API_KEY"] = args.api_key
        CONFIG["anthropic_api_key"] = args.api_key
        logger.info("명령줄 인수로 제공된 API 키를 사용합니다.")
    
    # 환경 변수로 설정
    if args.embed_dir != CONFIG["embed_dir"]:
        os.environ["EMBED_DIR"] = args.embed_dir
        CONFIG["embed_dir"] = args.embed_dir
        CONFIG["index_path"] = os.path.join(args.embed_dir, "faiss_index.bin")
        CONFIG["docs_path"] = os.path.join(args.embed_dir, "chunked_docs.pkl")
    
    # 서버 실행
    run_server(
        host=args.host,
        port=args.port,
        data_dir=args.data_dir,
        embed_dir=args.embed_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        force_embedding=args.force_embedding
    )