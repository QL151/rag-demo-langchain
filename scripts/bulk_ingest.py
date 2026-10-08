"""批量入库：把一整个文件夹的文档灌进知识库（走增量入库，可指定部门）。

用法（在项目根目录下执行）：

    # 1) 先看看会导入哪些文件，不真的入库
    python scripts/bulk_ingest.py "D:\\你的\\文档目录" --dry-run

    # 2) 真的入库（打上部门标签，便于检索时做数据隔离）
    python scripts/bulk_ingest.py "D:\\你的\\文档目录" --dept engineering

    # 3) 不带参数时，默认读项目里的 data/corpus 目录
    python scripts/bulk_ingest.py

    # 4) 换了 embedding 模型后，强制全量重建向量
    python scripts/bulk_ingest.py "D:\\你的\\文档目录" --force

说明：
- 递归扫描支持的类型（txt / md / csv / json / pdf），跳过内容过短的文件；
- **走的是 RagService 的增量入库**：文档指纹没变 → 整篇跳过；
  变了 → 只重新向量化"文字变了"的片段。所以重复跑这个脚本是安全的、便宜的；
- 批量导入时先不落盘（persist=False），最后统一 flush 一次，
  避免内置向量库"每篇都重写整个索引文件"造成的 O(n²) 磁盘写入；
- 单个文件出错不会中断整体，会记录下来继续跑。

⚠️ 重要：**先停掉正在运行的服务，再跑本脚本。**
   内置向量库是"整个文件重写"的。如果服务正在运行，它内存里还持有旧索引，
   之后你在网页上做任何操作（上传/删除文档）都会把旧索引写回去，**覆盖掉本次导入**。
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chat_store import ChatStore  # noqa: E402
from app.config import BASE_DIR, settings  # noqa: E402
from app.llm import EmbeddingUnavailable, LLMClient  # noqa: E402
from app.loaders import UnsupportedFile, read_text  # noqa: E402
from app.rag import RagService  # noqa: E402
from app.vectorstore import build_store  # noqa: E402

SUFFIXES = {".txt", ".md", ".markdown", ".csv", ".json", ".pdf"}
MIN_CHARS = 50

# 文件名以 _ 或 . 开头的，通常是元数据/隐藏文件（比如 _manifest.json），
# 它们不是知识库内容，导进去只会污染检索结果
IGNORE_PREFIXES = ("_", ".")


def should_ignore(path: Path) -> bool:
    return path.name.startswith(IGNORE_PREFIXES)


def flag_value(name: str) -> str | None:
    if name in sys.argv:
        try:
            return sys.argv[sys.argv.index(name) + 1]
        except IndexError:
            return None
    return None


def warn_if_server_running(force: bool = False) -> None:
    """检测 8000 端口是否有服务在跑。有的话提醒用户先停掉，避免索引被覆盖。"""
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=2) as resp:
            if resp.status != 200:
                return
    except (urllib.error.URLError, OSError):
        return  # 连不上 = 服务没在跑，正常

    print("!" * 62)
    print("[!] 检测到 8000 端口有服务正在运行")
    print("    内置向量库的索引是整体重写的文件。运行中的服务内存里还持有旧索引，")
    print("    它之后一旦保存（你在网页上传/删除文档时），就会覆盖掉本次导入的内容。")
    print("    建议：先停掉服务（终端按 Ctrl+C，或点 PyCharm 的红色停止按钮）。")
    print("!" * 62)
    if force:
        print("[--force] 已跳过确认，继续导入。")
        print("[--force] 导入完成后请务必：停掉服务 → 重新启动，让它加载新索引。")
        return
    try:
        answer = input("仍要继续吗？(y/N) ").strip().lower()
    except EOFError:  # 非交互环境（比如被脚本调用），安全起见直接取消
        answer = ""
    if answer not in {"y", "yes"}:
        print("已取消。请先停掉服务再运行（或加 --force 强制导入）。")
        raise SystemExit(1)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--") and not _is_flag_value(a)]
    dry_run = "--dry-run" in sys.argv
    force = "--force" in sys.argv
    dept = flag_value("--dept") or settings.default_dept
    doc_type = flag_value("--doc-type") or ""
    target = Path(args[0]) if args else (BASE_DIR / "data" / "corpus")

    if not target.exists():
        print(f"[×] 目录不存在：{target}")
        print('    用法：python scripts/bulk_ingest.py "D:\\你的\\文档目录" --dept 部门名')
        return 1

    files = [p for p in sorted(target.rglob("*")) if p.is_file() and p.suffix.lower() in SUFFIXES]
    ignored = [p for p in files if should_ignore(p)]
    files = [p for p in files if not should_ignore(p)]
    print(f"目录：{target}")
    print(f"发现 {len(files)} 个候选文件" + (f"（另有 {len(ignored)} 个以下划线/点开头的文件已忽略）" if ignored else ""))
    print(f"部门标签：{dept}" + (f"｜文档类型：{doc_type}" if doc_type else ""))
    print()

    if not files:
        print("[×] 这个目录里没有支持的文件类型")
        return 1

    if dry_run:
        for path in files[:60]:
            size = path.stat().st_size
            print(f"    {path.name}  ({size / 1024:.1f}KB)")
        if len(files) > 60:
            print(f"    ...（还有 {len(files) - 60} 个）")
        print("\n（--dry-run：只是预览，没有真正入库）")
        return 0

    settings.ensure_dirs()
    warn_if_server_running(force=force)
    llm = LLMClient(settings)
    store = build_store(settings)
    rag = RagService(settings, llm, store, ChatStore(settings.db_path))
    print(f"向量库：{store.backend}；向量化：{llm.describe()['embedding_model']}\n")

    ok = skipped_unchanged = 0
    embedded_chunks = reused_chunks = 0
    failed: list[str] = []
    started = time.time()

    for i, path in enumerate(files, 1):
        # 用相对路径做 source，避免不同子目录里同名文件互相覆盖
        source = path.name if path.parent == target else str(path.relative_to(target)).replace("\\", "/")
        try:
            text = read_text(path)
        except UnsupportedFile as exc:
            failed.append(f"{source}（{exc}）")
            continue
        if len(text.strip()) < MIN_CHARS:
            failed.append(f"{source}（内容过短，已跳过）")
            continue

        try:
            # persist=False：批量导入先不落盘，最后统一 flush 一次
            result = rag.ingest_text(
                source, text, dept=dept, doc_type=doc_type, force=force, persist=False
            )
        except EmbeddingUnavailable as exc:
            print(f"\n[×] 向量化失败，已中止（不继续污染索引）：\n{exc}")
            return 1
        except Exception as exc:  # noqa: BLE001 —— 单个文件出错不中断整体
            failed.append(f"{source}（{type(exc).__name__}: {exc}）")
            continue

        if result["skipped"]:
            skipped_unchanged += 1
        else:
            ok += 1
            embedded_chunks += result["embedded_chunks"]
            reused_chunks += result["reused_chunks"]

        if i % 10 == 0 or i == len(files):
            print(f"  已处理 {i}/{len(files)}（新入库 {ok}｜未变化跳过 {skipped_unchanged}）")

    store.flush()   # 统一落盘

    print("\n" + "=" * 66)
    print(f"[√] 入库完成：{ok} 个文件")
    print(f"    未变化整篇跳过：{skipped_unchanged} 个（增量入库省下的部分）")
    print(f"    片段：复用旧向量 {reused_chunks} 段｜真正重新向量化 {embedded_chunks} 段")
    print(f"    跳过/失败：{len(failed)} 个")
    for item in failed[:10]:
        print("      -", item)
    if len(failed) > 10:
        print(f"      ...（还有 {len(failed) - 10} 个）")
    print(f"    总用时：{time.time() - started:.0f}s")
    print(f"    当前知识库：{len(store.sources())} 篇文档 / {store.count()} 段")
    print("=" * 66)
    print("接下来：运行 run.py 或 uvicorn app.main:app --port 8000，打开页面提问")
    return 0


def _is_flag_value(value: str) -> bool:
    """判断某个位置参数是不是 --dept / --doc-type 的值（别把它当成目录路径）。"""
    flags = {"--dept", "--doc-type"}
    argv = sys.argv[1:]
    for index, item in enumerate(argv):
        if item in flags and index + 1 < len(argv) and argv[index + 1] == value:
            return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
