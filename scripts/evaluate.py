"""检索命中率快速评测：跑一遍评测集，算出 Top-1 / Top-4 命中率。

用法：
    python scripts/evaluate.py                 # 用已入库的知识库
    python scripts/evaluate.py --dept hr       # 只测某个部门的数据

评测集从 `data/sample/评测集.md` 读（**单一数据源**）。

为什么改成读文件：以前评测集在脚本里硬编码了一份，markdown 里还有一份，
改一处忘另一处就会漂移 —— 于是"评测通过"和"文档里写的用例"根本不是一回事。
这类"双份真相"是评测体系里最隐蔽的坑。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chat_store import ChatStore  # noqa: E402
from app.config import BASE_DIR, settings  # noqa: E402
from app.llm import LLMClient  # noqa: E402
from app.rag import RagService  # noqa: E402
from app.vectorstore import build_store  # noqa: E402

EVAL_SET = BASE_DIR / "data" / "sample" / "评测集.md"


def load_cases(path: Path) -> list[tuple[str, str]]:
    """从 markdown 表格里读「问题 | 期望关键词」，跳过表头和分隔行。"""
    cases: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2 or set(cells[0]) <= set("-: ") or cells[0] == "问题":
            continue
        cases.append((cells[0], cells[1]))
    return cases


def main() -> int:
    if not EVAL_SET.exists():
        print(f"[×] 找不到评测集：{EVAL_SET}")
        return 1
    cases = load_cases(EVAL_SET)
    if not cases:
        print(f"[×] 评测集是空的：{EVAL_SET}")
        return 1

    dept = None
    if "--dept" in sys.argv:
        try:
            dept = sys.argv[sys.argv.index("--dept") + 1]
        except IndexError:
            dept = None

    settings.ensure_dirs()
    rag = RagService(
        settings, LLMClient(settings), build_store(settings), ChatStore(settings.db_path)
    )

    if rag.store.count() == 0:
        print("[×] 知识库是空的。请先上传 data/sample/产品手册示例.md，或批量导入你的文档")
        return 1

    print(f"评测集：{EVAL_SET.name}（{len(cases)} 条）" + (f"｜部门过滤：{dept}" if dept else ""))
    print(f"检索配置：混合={settings.hybrid_enabled}｜精排={settings.rerank_ready}｜"
          f"阈值={settings.min_rerank_score if settings.rerank_ready else settings.min_vector_score}")
    print("-" * 74)
    print(f"{'问题':<26} {'Top-1':<7} {'Top-4':<7} 最高分  命中通道")
    print("-" * 74)

    top1 = top4 = refused = 0
    for question, keyword in cases:
        result = rag.search(question, 4, dept=dept)
        selected = rag._select(result)
        texts = [candidate.text for candidate in selected]
        hit1 = bool(texts) and keyword in texts[0]
        hit4 = any(keyword in text for text in texts)
        top1 += hit1
        top4 += hit4
        refused += result.refused
        top = selected[0] if selected else None
        print(
            f"{question:<26} {'✅' if hit1 else '❌':<7} {'✅' if hit4 else '❌':<7} "
            f"{(top.final_score if top else 0):<8} {'+'.join(top.matched_by) if top else '拒答'}"
        )

    total = len(cases)
    print("-" * 74)
    print(f"Top-1 命中率：{top1}/{total} = {top1 / total:.0%}")
    print(f"Top-4 命中率：{top4}/{total} = {top4 / total:.0%}")
    print(f"拒答条数：{refused}/{total}（阈值判定为'资料中没有依据'）")
    print("\n提示：命不中的话先看「命中通道」——只出现在『关键词』一列，")
    print("      说明向量模型对这个表述不敏感；两列都空说明该片段压根没被召回。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
