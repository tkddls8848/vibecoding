#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
claude_rag.py: 통합 Claude RAG 시스템 코어 모듈

이 모듈은 다음 기능들을 통합합니다:
- 환경 설정 및 구성 관리 (config.py)
- 임베딩 모델 초기화 및 관리 (model.py)
- 문서 임베딩 생성 (embedding.py)
- 쿼리 기반 문서 검색 (retriever.py)
- Claude API 통합 (claude_rag_api.py)
- 시스템 프롬프트 관리 (prompt.py)
"""

import os
import glob
import json
import pickle
import numpy as np
import logging
import faiss
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("claude_rag")

#
# 환경 설정 부분 (원래 config.py)
#
def load_config() -> Dict[str, Any]:
    """
    환경 변수 및 설정 로드
    
    Returns:
        설정 정보가 담긴 딕셔너리
    """
    # 현재 파일 위치 기준으로 .env 파일 로드
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path=dotenv_path, override=True)
    
    config = {
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "openapi_key": os.getenv("OPENAPI_KEY"),  # 조달청 OpenAPI serviceKey
        "model_name": os.getenv("MODEL_NAME", "claude-3-7-sonnet-20250219"),
        "temperature": float(os.getenv("TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("MAX_TOKENS", "1000")),
        "embed_dir": os.getenv("EMBED_DIR", "./embeddings"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "500")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "100")),
        "top_k": int(os.getenv("TOP_K", "5")),
        "supported_extensions": os.getenv("SUPPORTED_EXTENSIONS", ".md,.txt").split(",")
    }
    
    # 파일 경로 설정
    config["index_path"] = os.path.join(config["embed_dir"], "faiss_index.bin")
    config["docs_path"] = os.path.join(config["embed_dir"], "chunked_docs.pkl")
    
    # API 키 확인
    if not config["anthropic_api_key"]:
        logger.warning("ANTHROPIC_API_KEY가 설정되지 않았습니다. API 호출이 실패할 수 있습니다.")
    
    if not config["openapi_key"]:
        logger.warning("OPENAPI_KEY가 설정되지 않았습니다. API URL 생성 시 인증 파라미터가 누락될 수 있습니다.")
    
    return config

# 전역 설정 객체
CONFIG = load_config()

#
# 임베딩 모델 부분 (원래 model.py)
#
class EmbeddingModel:
    """임베딩 모델을 싱글톤 패턴으로 관리하는 클래스"""
    
    _instance = None
    _model = None
    _dimension = None
    
    @classmethod
    def get_instance(cls):
        """임베딩 모델의 싱글톤 인스턴스 반환"""
        if cls._model is None:
            model_name = CONFIG["embedding_model"]
            logger.info(f"임베딩 모델 초기화: {model_name}")
            cls._model = SentenceTransformer(model_name)
            cls._dimension = cls._model.get_sentence_embedding_dimension()
            
        return cls._model
    
    @classmethod
    def get_dimension(cls):
        """임베딩 모델의 차원 반환"""
        if cls._dimension is None:
            # 모델 인스턴스 확보
            cls.get_instance()
        return cls._dimension

#
# 문서 임베딩 부분 (원래 embedding.py)
#
class DocumentEmbedder:
    def __init__(self):
        """임베딩 모델 초기화"""
        # 싱글톤 임베딩 모델 사용
        self.model = EmbeddingModel.get_instance()
        self.dimension = EmbeddingModel.get_dimension()
    
    def _read_text_files(self, directory: str) -> List[Dict[str, Any]]:
        """디렉토리에서 텍스트 파일들을 읽어오는 함수 (마크다운 및 텍스트 파일 지원)"""
        documents = []
        
        # 지원하는 확장자들
        supported_extensions = CONFIG["supported_extensions"]
        logger.info(f"지원하는 파일 확장자: {supported_extensions}")
        
        # 각 확장자별로 파일 수집
        all_files = []
        for ext in supported_extensions:
            ext = ext.strip()  # 공백 제거
            if not ext.startswith('.'):
                ext = '.' + ext
            pattern = os.path.join(directory, f"*{ext}")
            files = glob.glob(pattern)
            all_files.extend(files)
            logger.info(f"{ext} 파일 {len(files)}개 발견")
        
        # 중복 제거 (같은 파일이 여러 확장자로 매칭될 수 있음)
        all_files = list(set(all_files))
        logger.info(f"총 {len(all_files)}개의 파일을 처리합니다.")
        
        for file_path in tqdm(all_files, desc="텍스트 파일 읽기"):
            try:
                # 파일 인코딩 자동 감지 시도
                encodings_to_try = ['utf-8', 'cp949', 'euc-kr', 'latin-1']
                content = None
                used_encoding = None
                
                for encoding in encodings_to_try:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read().strip()
                        used_encoding = encoding
                        break
                    except UnicodeDecodeError:
                        continue
                
                if content is None:
                    logger.error(f"파일 인코딩 감지 실패: {file_path}")
                    continue
                
                # 빈 파일 건너뛰기
                if not content:
                    logger.warning(f"빈 파일 건너뜀: {file_path}")
                    continue
                
                # 문서를 내용, 파일 이름, 경로, 확장자 등으로 구성된 딕셔너리로 저장
                file_name = os.path.basename(file_path)
                file_ext = os.path.splitext(file_path)[1].lower()
                
                doc = {
                    "content": content,
                    "file_name": file_name,
                    "file_path": file_path,
                    "file_extension": file_ext,
                    "encoding_used": used_encoding
                }
                documents.append(doc)
                
            except Exception as e:
                logger.error(f"파일 읽기 오류 ({file_path}): {str(e)}")
        
        return documents
    
    def _chunk_documents(self, documents: List[Dict[str, Any]], chunk_size: int = None, overlap: int = None) -> List[Dict[str, Any]]:
        """문서를 청크로 분할하는 함수"""
        # 환경 설정에서 값을 가져오거나 기본값 사용
        chunk_size = chunk_size or CONFIG["chunk_size"]
        overlap = overlap or CONFIG["chunk_overlap"]
        
        chunked_docs = []
        
        for doc in tqdm(documents, desc="문서 청킹"):
            content = doc["content"]
            
            if len(content) <= chunk_size:
                # 문서가 청크 크기보다 작으면 그대로 사용
                chunk_doc = doc.copy()
                chunk_doc["chunk_id"] = 0
                chunked_docs.append(chunk_doc)
            else:
                # 청크 크기로 분할
                for i in range(0, len(content), chunk_size - overlap):
                    if i > 0:
                        start = i
                    else:
                        start = 0
                    
                    end = min(start + chunk_size, len(content))
                    chunk_text = content[start:end]
                    
                    if len(chunk_text) < 50:  # 너무 작은 청크는 건너뜀
                        continue
                    
                    chunk_doc = doc.copy()
                    chunk_doc["content"] = chunk_text
                    chunk_doc["chunk_id"] = i // (chunk_size - overlap)
                    chunked_docs.append(chunk_doc)
        
        return chunked_docs
    
    def create_embeddings(self, data_dir: str, output_dir: str = None, chunk_size: int = None, overlap: int = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        데이터 디렉토리의 텍스트 파일들을 임베딩하고 결과를 저장
        
        Args:
            data_dir: 텍스트 파일이 있는 디렉토리 경로
            output_dir: 임베딩 결과를 저장할 디렉토리 경로
            chunk_size: 청크 크기 (문자 수)
            overlap: 청크 간 겹치는 영역 크기 (문자 수)
            
        Returns:
            embeddings: 임베딩 벡터 (numpy 배열)
            chunked_docs: 청크 문서 리스트
        """
        # 환경 설정에서 값을 가져오거나 기본값 사용
        output_dir = output_dir or CONFIG["embed_dir"]
        chunk_size = chunk_size or CONFIG["chunk_size"]
        overlap = overlap or CONFIG["chunk_overlap"]
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"1. 텍스트 파일 읽기: {data_dir}")
        documents = self._read_text_files(data_dir)
        logger.info(f"   - 총 {len(documents)}개의 문서를 읽었습니다.")
        
        if not documents:
            logger.error(f"처리할 문서가 없습니다. 지원하는 파일 형식: {CONFIG['supported_extensions']}")
            raise ValueError("처리할 문서가 없습니다.")
        
        logger.info(f"2. 문서를 청크로 분할 (크기: {chunk_size}, 겹침: {overlap})")
        chunked_docs = self._chunk_documents(documents, chunk_size, overlap)
        logger.info(f"   - 총 {len(chunked_docs)}개의 청크가 생성되었습니다.")
        
        logger.info("3. 청크 임베딩 생성")
        texts = [doc["content"] for doc in chunked_docs]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        logger.info(f"   - 임베딩 차원: {embeddings.shape}")
        
        logger.info("4. 결과 저장")
        # FAISS 인덱스 생성 및 저장
        index = faiss.IndexFlatL2(self.dimension)
        faiss.normalize_L2(embeddings)  # L2 정규화
        index.add(embeddings.astype('float32'))
        
        # FAISS 인덱스 저장
        faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
        
        # 문서 청크 저장 - Python 3.12에서의 호환성을 위해 프로토콜 버전 명시
        with open(os.path.join(output_dir, "chunked_docs.pkl"), "wb") as f:
            pickle.dump(chunked_docs, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 임베딩 저장
        with open(os.path.join(output_dir, "embeddings.npy"), "wb") as f:
            np.save(f, embeddings)
        
        logger.info(f"5. 임베딩 완료: 결과는 {output_dir}에 저장되었습니다.")
        return embeddings, chunked_docs

#
# 문서 검색 부분 (원래 retriever.py)
#
class DocumentRetriever:
    def __init__(self, index_path: str = None, docs_path: str = None):
        """
        문서 검색 클래스 초기화
        
        Args:
            index_path: FAISS 인덱스 파일 경로
            docs_path: 청크 문서 데이터 파일 경로
        """
        # 환경 설정에서 값을 가져오거나 매개변수 사용
        self.index_path = index_path or CONFIG["index_path"]
        self.docs_path = docs_path or CONFIG["docs_path"]
        
        # 싱글톤 임베딩 모델 사용
        self.model = EmbeddingModel.get_instance()
        
        # FAISS 인덱스 로드
        try:
            self.index = faiss.read_index(self.index_path)
            logger.info(f"FAISS 인덱스 로드 완료: {self.index_path}")
        except Exception as e:
            logger.error(f"FAISS 인덱스 로드 오류: {str(e)}")
            raise
        
        # 청크 문서 데이터 로드
        try:
            with open(self.docs_path, "rb") as f:
                self.documents = pickle.load(f)
            logger.info(f"문서 데이터 로드 완료: {len(self.documents)}개 청크")
        except Exception as e:
            logger.error(f"문서 데이터 로드 오류: {str(e)}")
            raise
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        쿼리에 관련된 문서를 검색
        
        Args:
            query: 검색 쿼리
            top_k: 검색할 상위 문서 수
            
        Returns:
            검색된 문서 리스트
        """
        # 환경 설정에서 값을 가져오거나 매개변수 사용
        top_k = top_k or CONFIG["top_k"]
        
        # 쿼리 임베딩 생성
        query_embedding = self.model.encode([query])[0]
        
        # L2 정규화
        faiss.normalize_L2(np.array([query_embedding], dtype='float32'))
        
        # FAISS로 검색
        D, I = self.index.search(np.array([query_embedding], dtype='float32'), top_k)
        
        # 검색 결과를 원본 문서와 매핑
        results = []
        for i, idx in enumerate(I[0]):
            if idx < len(self.documents):  # 유효한 인덱스인지 확인
                doc = self.documents[idx].copy()
                doc["score"] = float(D[0][i])  # 유사도 점수 추가
                results.append(doc)
        
        logger.info(f"'{query}' 검색 결과: {len(results)}개 문서")
        return results
    
    def format_results_for_llm(self, results: List[Dict[str, Any]], max_tokens: int = 3000) -> str:
        """
        검색 결과를 LLM 입력용으로 포맷팅
        
        Args:
            results: 검색 결과 문서 리스트
            max_tokens: 최대 토큰 수
            
        Returns:
            LLM 입력용 포맷팅된 문자열
        """
        formatted_text = "다음은 참고 문서들입니다:\n\n"
        
        for i, doc in enumerate(results, 1):
            doc_text = f"[문서 {i}] {doc['file_name']}"
            if 'file_extension' in doc:
                doc_text += f" ({doc['file_extension']} 파일)"
            doc_text += "\n"
            doc_text += doc["content"]
            doc_text += f"\n\n"
            
            # 너무 길어지면 자르기
            if len(formatted_text + doc_text) > max_tokens:
                formatted_text += f"...(최대 토큰 제한으로 {len(results) - i + 1}개 문서 생략)..."
                break
                
            formatted_text += doc_text
            
        return formatted_text

# 프롬프트 모듈 임포트
from prompt import load_prompt

#
# Claude RAG API 부분 (원래 claude_rag_api.py)
#
class ClaudeRAG:
    def __init__(
        self, 
        index_path: str = None, 
        docs_path: str = None, 
        model_name: str = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        """
        Claude RAG 클래스 초기화
        
        Args:
            index_path: FAISS 인덱스 파일 경로
            docs_path: 청크 문서 데이터 파일 경로
            model_name: Claude 모델 이름
            temperature: 생성 다양성 (0~1)
            max_tokens: 최대 생성 토큰 수
        """
        # 환경 설정에서 값을 가져오거나 매개변수 사용
        self.model_name = model_name or CONFIG["model_name"]
        self.temperature = temperature or CONFIG["temperature"]
        self.max_tokens = max_tokens or CONFIG["max_tokens"]
        
        # API 키 설정
        api_key = CONFIG["anthropic_api_key"]
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        # Claude 클라이언트 초기화
        self.anthropic = Anthropic(api_key=api_key)
        
        # 시스템 프롬프트
        self.system_prompt = load_prompt(service_key=CONFIG.get("openapi_key", ""))
        
        # Retriever 초기화
        self.retriever = DocumentRetriever(
            index_path or CONFIG["index_path"], 
            docs_path or CONFIG["docs_path"]
        )
        
        logger.info(f"Claude RAG 시스템이 초기화되었습니다. 모델: {self.model_name}")
    
    async def query(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        쿼리에 대한 RAG 기반 답변 생성
        
        Args:
            query: 사용자 쿼리
            top_k: 검색할 상위 문서 수
            
        Returns:
            응답 결과 딕셔너리
        """
        try:
            # 환경 설정에서 값을 가져오거나 매개변수 사용
            top_k = top_k or CONFIG["top_k"]
            
            logger.info(f"쿼리 처리 중: '{query}'")
            
            # 1. 관련 문서 검색
            logger.info(f"관련 문서 검색 (top_k = {top_k})...")
            search_results = self.retriever.search(query, top_k)
            
            # 2. 검색 결과 포맷팅
            formatted_context = self.retriever.format_results_for_llm(search_results)
            logger.info(f"검색 결과 포맷팅 완료 (길이: {len(formatted_context)} 자)")
            
            # 3. 최종 사용자 프롬프트 구성
            user_prompt = f"{formatted_context}\n\n위 문서들을 참고하여 다음 질문에 답해주세요: {query}"
            logger.info(f"Claude API 호출 중...")
            
            # 4. Claude API 호출
            response = self.anthropic.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )
            
            # 응답 추출
            claude_response = response.content[0].text
            
            logger.info(f"Claude API 응답 수신 완료")
            
            # 5. 결과 구성
            used_docs = []
            for doc in search_results:
                used_docs.append({
                    "file_name": doc["file_name"],
                    "score": doc["score"],
                    "file_extension": doc.get("file_extension", "unknown"),
                    "preview": doc["content"][:150] + "..." if len(doc["content"]) > 150 else doc["content"]
                })
            
            result = {
                "query": query,
                "results": [
                    {
                        "type": "claude_rag_response",
                        "content": claude_response,
                        "model": self.model_name,
                        "used_documents": used_docs
                    }
                ],
                "status": "success"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"쿼리 처리 오류: {str(e)}")
            raise Exception(f"Claude RAG API 호출 중 오류 발생: {str(e)}")

async def query_claude_rag(
    query: str,
    index_path: str = None,
    docs_path: str = None,
    model_name: str = None,
    temperature: float = None,
    max_tokens: int = None,
    top_k: int = None
) -> Dict[str, Any]:
    """
    Claude RAG API 래퍼 함수
    
    Args:
        query: 사용자 쿼리
        index_path: FAISS 인덱스 파일 경로
        docs_path: 청크 문서 데이터 파일 경로
        model_name: Claude 모델 이름
        temperature: 생성 다양성 (0~1)
        max_tokens: 최대 생성 토큰 수
        top_k: 검색할 상위 문서 수
        
    Returns:
        응답 결과 딕셔너리
    """
    claude_rag = ClaudeRAG(
        index_path=index_path,
        docs_path=docs_path,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return await claude_rag.query(query, top_k)


if __name__ == "__main__":
    # 스크립트 직접 실행 시
    import argparse
    
    parser = argparse.ArgumentParser(description="문서 임베딩 및 검색 도구")
    parser.add_argument("--embed", action="store_true", help="문서 임베딩 생성")
    parser.add_argument("--search", type=str, help="검색 쿼리")
    parser.add_argument("--data_dir", type=str, default="../data", help="텍스트 파일이 있는 디렉토리")
    parser.add_argument("--output_dir", type=str, default=CONFIG["embed_dir"], help="임베딩 결과를 저장할 디렉토리")
    parser.add_argument("--chunk_size", type=int, default=CONFIG["chunk_size"], help="청크 크기 (문자 수)")
    parser.add_argument("--overlap", type=int, default=CONFIG["chunk_overlap"], help="청크 간 겹치는 영역 크기")
    parser.add_argument("--top_k", type=int, default=CONFIG["top_k"], help="검색할 상위 문서 수")
    
    args = parser.parse_args()
    
    if args.embed:
        # 임베딩 생성
        embedder = DocumentEmbedder()
        embedder.create_embeddings(
            args.data_dir, 
            args.output_dir, 
            args.chunk_size, 
            args.overlap
        )
    
    if args.search:
        # 문서 검색
        retriever = DocumentRetriever()
        results = retriever.search(args.search, args.top_k)
        
        print(f"\n=== '{args.search}'에 대한 검색 결과 ===")
        for i, doc in enumerate(results, 1):
            file_ext = doc.get('file_extension', 'unknown')
            print(f"\n[{i}] 점수: {doc['score']:.4f}, 파일: {doc['file_name']} ({file_ext})")
            print(f"내용 미리보기: {doc['content'][:150]}...")