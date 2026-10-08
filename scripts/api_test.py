"""HTTP 接口自测（需要先启动服务）：

    终端 A：uvicorn app.main:app --port 8000
    终端 B：python scripts/api_test.py

只用标准库 urllib，不需要额外安装依赖。会依次验证：
/api/health → /api/documents(上传) → /api/documents(列表) → /api/chat(search)
→ /api/chat(问答，含缓存命中) → /api/stats → /api/traces
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE_URL = next((arg for arg in sys.argv[1:] if arg.startswith("http")), "http://127.0.0.1:8000")
ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample" / "产品手册示例.md"

FAILURES: list[str] = []


def request(method: str, path: str, body: bytes | None = None, content_type: str | None = None):
    req = urllib.request.Request(BASE_URL + path, data=body, method=method)
    if content_type:
        req.add_header("Content-Type", content_type)
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else None


def multipart(filename: str, content: bytes) -> tuple[bytes, str]:
    boundary = "----ragdemo" + uuid.uuid4().hex
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: text/markdown\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return head + content + tail, f"multipart/form-data; boundary={boundary}"


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"    [√] {message}")
    else:
        print(f"    [×] {message}")
        FAILURES.append(message)


def main() -> int:
    try:
        health = request("GET", "/api/health")
    except urllib.error.URLError as exc:
        print(f"[×] 连不上 {BASE_URL}，请先运行：uvicorn app.main:app --port 8000\n    错误：{exc}")
        return 1

    print("[1] GET /api/health")
    print(f"    向量库={health['vector_backend']}｜文档={health['documents']}｜片段={health['chunks']}")
    print(f"    检索配置={json.dumps(health['retrieval'], ensure_ascii=False)}")
    check(health["status"] == "ok", "服务在线")
    for warning in health.get("warnings", []):
        print(f"    [!] {warning}")

    print("[2] POST /api/documents（上传并入库，走增量）")
    body, content_type = multipart(SAMPLE.name, SAMPLE.read_bytes())
    uploaded = request("POST", "/api/documents", body, content_type)
    print(f"    {json.dumps(uploaded, ensure_ascii=False)}")
    check(uploaded["chunks"] > 0, "切分出了片段")
    check(
        uploaded["skipped"] or uploaded["embedded_chunks"] > 0,
        "要么新入库、要么整篇跳过（增量入库生效）",
    )

    print("[3] POST /api/documents（重复上传同一份 → 应当整篇跳过）")
    body, content_type = multipart(SAMPLE.name, SAMPLE.read_bytes())
    again = request("POST", "/api/documents", body, content_type)
    check(again["skipped"] is True, f"第二次上传被跳过（embedded_chunks={again['embedded_chunks']}）")

    print("[4] GET /api/documents")
    documents = request("GET", "/api/documents")
    print(f"    {json.dumps(documents[:3], ensure_ascii=False)}")
    check(isinstance(documents, list) and documents, "文档列表非空")

    print("[5] POST /api/chat mode=search（只检索，不调用大模型）")
    payload = json.dumps({"question": "专业版多少钱？", "session_id": "api-test", "mode": "search"}).encode()
    search = request("POST", "/api/chat", payload, "application/json")
    for citation in search["citations"]:
        print(
            f"    引用[{citation['index']}] {citation['source']} "
            f"最终={citation['score']} 向量={citation['vector_score']} "
            f"精排={citation['rerank_score']} 通道={citation['matched_by']}"
        )
    print(f"    {search['answer']}")
    check(bool(search["citations"]), "检索到了引用")

    print("[6] POST /api/chat（完整问答，第一次应为 miss，第二次应为 exact 缓存命中）")
    # 先清空答案缓存：这一步是可复现的前提，否则上一次跑的结果会让"第一次"也命中缓存。
    # 顺便也验证了 DELETE /api/cache。
    cleared = request("DELETE", "/api/cache")
    print(f"    已清空答案缓存：{cleared['deleted']} 条")
    payload = json.dumps({"question": "专业版多少钱？", "session_id": "api-test"}).encode()
    first = request("POST", "/api/chat", payload, "application/json")
    second = request("POST", "/api/chat", payload, "application/json")
    print(f"    第一次 cache={first['cache']}｜耗时 {first['latency_ms']}ms"
          f"｜tokens={first['usage']['prompt_tokens']}+{first['usage']['completion_tokens']}"
          f"｜费用 {first['usage']['cost_yuan']} 元")
    print(f"    第二次 cache={second['cache']}｜耗时 {second['latency_ms']}ms")
    check(first["cache"] == "miss", "第一次未命中缓存")
    check(second["cache"] in {"exact", "semantic"}, "第二次命中缓存")
    print("    回答：", first["answer"].replace("\n", "\n          ")[:400])

    print("[7] POST /api/chat（知识库里没有的问题 → 应当拒答且不编造）")
    payload = json.dumps({"question": "请问怎么养一只柯基犬", "session_id": "api-test"}).encode()
    refused = request("POST", "/api/chat", payload, "application/json")
    print(f"    refused={refused['refused']}｜citations={len(refused['citations'])}")
    print(f"    {refused['answer'][:120]}")
    check(refused["refused"] is True, "低置信度问题被拒答")

    print("[8] GET /api/traces/{trace_id}")
    trace = request("GET", f"/api/traces/{first['trace_id']}")
    stages = " → ".join(f"{stage['stage']}({stage['ms']}ms)" for stage in trace["stages"])
    print(f"    链路：{stages}")
    check(any(stage["stage"] == "retrieve" for stage in trace["stages"]), "trace 记录了检索阶段")

    print("[9] GET /api/stats")
    stats = request("GET", "/api/stats")
    print(f"    用量：{json.dumps(stats['usage'], ensure_ascii=False)[:220]}")
    print(f"    缓存：{json.dumps(stats['cache'], ensure_ascii=False)}")
    check(stats["usage"]["calls"] >= 2, "用量有记录")
    check(stats["knowledge"]["chunks"] > 0, "知识库统计正常")

    print()
    if FAILURES:
        print(f"[×] 有 {len(FAILURES)} 项没通过：")
        for item in FAILURES:
            print("    -", item)
        return 1
    print("[√] 接口自测全部通过。SSE 流式接口请用浏览器页面验证：", BASE_URL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
