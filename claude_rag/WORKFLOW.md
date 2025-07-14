# Claude RAG 시스템 (JSON/XML/MD 지원)

조달청 나라장터 API 문서 기반의 Claude RAG(Retrieval-Augmented Generation) 시스템입니다. 이제 **JSON, XML, Markdown** 파일 형식을 지원합니다.

## 주요 특징

- 🔍 **다양한 파일 형식 지원**: JSON, XML, Markdown (.md) 파일 임베딩
- 🤖 **Claude API 통합**: Anthropic Claude를 사용한 고품질 답변 생성
- 📊 **벡터 검색**: FAISS를 이용한 빠른 문서 검색
- 🌐 **웹 인터페이스**: 사용하기 쉬운 웹 UI 제공
- 🔧 **API 엔드포인트**: RESTful API 지원

## 지원하는 파일 형식

### 1. JSON 파일 (.json)
- 모든 문자열 값을 재귀적으로 추출
- 키-값 쌍 형태로 구조화된 데이터 처리
- API 응답, 설정 파일, 데이터 등에 적합

### 2. XML 파일 (.xml)
- 모든 텍스트 노드와 속성 값 추출
- 계층 구조 데이터 처리
- API 명세서, 설정 파일 등에 적합

### 3. Markdown 파일 (.md, .markdown)
- HTML 변환 후 순수 텍스트 추출
- 문서화, README, 가이드 등에 적합

## 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 추가:

```env
# 필수 설정
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# 선택적 설정
OPENAPI_KEY=your_openapi_service_key_here
MODEL_NAME=claude-3-sonnet-20240229
TEMPERATURE=0.7
MAX_TOKENS=1000
EMBED_DIR=./embeddings
EMBEDDING_MODEL=jhgan/ko-sroberta-multitask
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K=5
```

### 3. 데이터 디렉토리 준비

지원하는 파일들을 데이터 디렉토리에 배치:

```
data/
├── api_docs.json      # JSON 형식 API 문서
├── specification.xml   # XML 형식 명세서
├── guide.md           # Markdown 가이드
└── readme.markdown    # Markdown 문서
```

## 사용법

### 1. 서버 실행

```bash
python server.py --data-dir ./data --host localhost --port 8001
```

옵션:
- `--data-dir`: JSON/XML/MD 파일이 있는 디렉토리
- `--host`: 서버 호스트 (기본값: localhost)
- `--port`: 서버 포트 (기본값: 8001)
- `--embed-dir`: 임베딩 저장 디렉토리
- `--chunk-size`: 문서 청크 크기
- `--overlap`: 청크 간 겹치는 영역 크기
- `--force-embedding`: 임베딩 강제 재생성
- `--api-key`: Anthropic API 키 (환경변수 대신)

### 2. 웹 인터페이스 사용

브라우저에서 `http://localhost:8001` 접속하여 질문 입력

### 3. API 엔드포인트 사용

#### GET 요청
```bash
curl "http://localhost:8001/query?q=입찰공고는%20어떻게%20조회하나요?"
```

#### POST 요청
```bash
curl -X POST "http://localhost:8001/query" \
     -H "Content-Type: application/json" \
     -d '{
       "q": "입찰공고는 어떻게 조회하나요?",
       "top_k": 5,
       "temperature": 0.7,
       "max_tokens": 1000
     }'
```

### 4. 명령줄에서 직접 사용

#### 임베딩 생성
```bash
python claude_rag.py --embed --data_dir ./data --chunk_size 500 --overlap 100
```

#### 문서 검색 테스트
```bash
python claude_rag.py --search "입찰공고 조회 방법" --top_k 5
```

## 파일 형식별 처리 방식

### JSON 파일 처리
```python
# 예시 JSON 구조
{
  "api_name": "입찰공고 조회",
  "endpoint": "/getBidPblancListInfo",
  "description": "입찰공고 목록을 조회합니다",
  "parameters": {
    "serviceKey": "인증키",
    "pageNo": "페이지 번호"
  }
}

# 추출되는 텍스트
api_name: 입찰공고 조회
endpoint: /getBidPblancListInfo
description: 입찰공고 목록을 조회합니다
serviceKey: 인증키
pageNo: 페이지 번호
```

### XML 파일 처리
```xml
<!-- 예시 XML 구조 -->
<api name="입찰공고조회">
  <endpoint>/getBidPblancListInfo</endpoint>
  <description>입찰공고 목록을 조회합니다</description>
  <parameter name="serviceKey" required="true">인증키</parameter>
</api>

<!-- 추출되는 텍스트 -->
name: 입찰공고조회
/getBidPblancListInfo
입찰공고 목록을 조회합니다
name: serviceKey
required: true
인증키
```

### Markdown 파일 처리
```markdown
# API 가이드

## 입찰공고 조회

**엔드포인트**: `/getBidPblancListInfo`

입찰공고 목록을 조회하는 API입니다.

### 파라미터
- serviceKey: 인증키 (필수)
- pageNo: 페이지 번호 (선택)
```

## 시스템 구조

```
claude_rag.py          # 메인 RAG 시스템 (JSON/XML/MD 지원)
├── DocumentEmbedder   # 문서 임베딩 생성 (다중 형식)
├── DocumentRetriever  # 문서 검색
├── ClaudeRAG         # Claude API 통합
└── EmbeddingModel    # 임베딩 모델 관리

server.py             # FastAPI 웹 서버
├── 웹 인터페이스     # HTML 템플릿
├── REST API         # JSON API 엔드포인트
└── 자동 임베딩 생성  # 서버 시작 시 임베딩 확인

prompt.py             # 시스템 프롬프트 관리
```

## 로그 및 디버깅

서버 실행 시 다음과 같은 로그를 확인할 수 있습니다:

```
INFO - 지원하는 파일 형식: JSON, XML, Markdown
INFO - 데이터 디렉토리에서 15개의 지원 파일을 찾았습니다.
INFO - JSON: 5개
INFO - XML: 7개  
INFO - Markdown: 3개
INFO - 총 342개의 청크가 생성되었습니다.
INFO - 임베딩 차원: (342, 768)
```

## 문제 해결

### 1. 지원하지 않는 파일 형식
```
❌ 데이터 디렉토리에 지원하는 파일(JSON/XML/MD)이 없습니다.
```
- 해결: 데이터 디렉토리에 .json, .xml, .md 파일을 추가

### 2. 파일 읽기 오류
```
ERROR - JSON 파일 처리 오류: Expecting value: line 1 column 1
```
- 해결: JSON 파일의 구문 오류 수정

### 3. XML 파싱 오류
```
ERROR - XML 파일 처리 오류: not well-formed
```
- 해결: XML 파일의 문법 오류 수정

### 4. 의존성 오류
```
ModuleNotFoundError: No module named 'beautifulsoup4'
```
- 해결: `pip install beautifulsoup4 lxml markdown`

## 성능 최적화

1. **청크 크기 조정**: 문서 특성에 맞게 `CHUNK_SIZE` 조정
2. **임베딩 모델 변경**: 더 적합한 한국어 모델 사용
3. **캐싱**: 자주 사용되는 쿼리 결과 캐싱
4. **배치 처리**: 대량 문서 처리 시 배치 크기 조정

## 라이선스

MIT License