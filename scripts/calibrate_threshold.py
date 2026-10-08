"""标定拒答阈值：用数据算出 MIN_VECTOR_SCORE 该设多少，而不是拍脑袋。

用法（项目根目录，需要知识库里已经有文档）：

    python scripts/calibrate_threshold.py
    python scripts/calibrate_threshold.py --probe 40     # 多测一些探针问题

它做什么：
    1. 准备一组**与语料无关**的探针问题（默认 24 条，如"怎么养柯基""明天下雨吗"）；
    2. 每条都去检索一遍，记下"最高向量分" —— 这就是"不该答的问题能拿到的最高分"；
    3. 再拿 `data/sample/评测集.md` 里**该答对的问题**跑一遍，记下最低分；
    4. 打印两条分布，并给出建议阈值。

为什么必须标定：
    向量相似度的绝对值高度依赖语料。真实实测（1734 个片段的语料）里，
    "怎么养一只柯基犬"这种完全无关的问题，最高分还能到 0.37 ——
    因为上千个片段里总能挑出一个"语义上沾点边"的。
    也就是说，拍脑袋定的 0.25 在这种语料上**一个候选都拦不住**。
    阈值必须是"测出来的"，不是"感觉出来的"。

不花钱：只调用 embedding 接口，几十条短文本，成本可以忽略。
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

# 与"企业知识库"天然无关的探针问题：跨领域、日常、具体。
# 用固定清单而不是让模型生成，是为了让这个脚本可复现、零额外成本。
PROBES = [
    "请问怎么养一只柯基犬？",
    "明天上海会下雨吗？",
    "红烧肉怎么做才软烂？",
    "世界杯上一届冠军是谁？",
    "我最近失眠该挂什么科？",
    "从北京到拉萨的火车要多久？",
    "钢琴初学者先练什么曲子？",
    "冰箱冷藏室结冰了怎么办？",
    "推荐几部适合周末看的悬疑电影",
    "汽车轮胎多久换一次？",
    "孩子几岁开始换牙？",
    "笔记本电脑电池怎么保养？",
    "仙人掌多久浇一次水？",
    "第一次去西藏要注意什么？",
    "怎么挑选一只健康的小猫？",
    "马拉松赛前一周该怎么吃？",
    "洗衣机不脱水是什么原因？",
    "阳台种番茄需要注意什么？",
    "怎么用空气炸锅烤鸡翅？",
    "哪个航空公司行李额最宽松？",
    "近视手术有哪几种？",
    "怎么把牛仔裤洗得不掉色？",
    "钓鱼新手要买什么装备？",
    "咖啡喝多了会怎么样？",
]

EVAL_SET = BASE_DIR / "data" / "sample" / "评测集.md"


def load_eval_questions(path: Path) -> list[str]:
    questions: list[str] = []
    if not path.exists():
        return questions
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2 or set(cells[0]) <= set("-: ") or cells[0] == "问题":
            continue
        questions.append(cells[0])
    return questions


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * ratio))
    return ordered[index]


def main() -> int:
    probe_count = 24
    if "--probe" in sys.argv:
        try:
            probe_count = max(4, int(sys.argv[sys.argv.index("--probe") + 1]))
        except (IndexError, ValueError):
            pass

    settings.ensure_dirs()
    llm = LLMClient(settings)
    store = build_store(settings)
    rag = RagService(settings, llm, store, ChatStore(settings.db_path))

    if store.count() == 0:
        print("[×] 知识库是空的，先导入文档再标定阈值")
        return 1

    probes = PROBES[:probe_count]
    print(f"知识库：{len(store.sources())} 篇 / {store.count()} 段")
    print(f"探针问题：{len(probes)} 条（与语料无关，理想情况下都该被拒答）")
    print(f"当前配置：MIN_VECTOR_SCORE={settings.min_vector_score}｜"
          f"MIN_RERANK_SCORE={settings.min_rerank_score}\n")

    print("正在向量化探针问题...")
    vectors = llm.embed(probes)

    # 直接查向量库、关掉阈值与精排：我们要的是"最高分分布"，不是最终判定结果
    negatives: list[float] = []
    print(f"\n{'探针问题':<26} 最高向量分")
    print("-" * 46)
    for question, vector in zip(probes, vectors):
        hits = store.query(vector, 1)
        score = hits[0].score if hits else 0.0
        negatives.append(score)
        print(f"{question:<26} {score}")

    print("-" * 46)
    print(
        f"无关问题的最高分：中位数 {percentile(negatives, 0.5):.3f}｜"
        f"P90 {percentile(negatives, 0.9):.3f}｜最大 {max(negatives):.3f}"
    )

    questions = load_eval_questions(EVAL_SET)
    positives: list[float] = []
    if questions:
        print("\n正在跑该答对的问题（来自 data/sample/评测集.md）...")
        for question, vector in zip(questions, llm.embed(questions)):
            hits = store.query(vector, 1)
            positives.append(hits[0].score if hits else 0.0)
        print(
            f"相关问题（应当答对）的最高分：最小 {min(positives):.3f}｜"
            f"中位数 {percentile(positives, 0.5):.3f}"
        )

    recommended = round(max(percentile(negatives, 0.9), 0.01) * 1.05, 2)
    print("\n" + "=" * 62)
    print(f"建议 MIN_VECTOR_SCORE = {recommended}")
    print("  依据：取无关问题最高分的 P90 再上浮 5%，这样约 90% 的无关问题会被拦下。")
    if positives and min(positives) < recommended:
        print(f"  [!] 注意：该答对的问题里有分数低到 {min(positives):.3f} 的，"
              f"阈值设到 {recommended} 会误伤它。")
        print("      这种情况说明「靠向量分拒答」已经不够用了，应该启用精排：")
        print("      cross-encoder（RERANK_PROVIDER=http）或 listwise（RERANK_PROVIDER=llm），")
        print("      让模型读完问题再决定要不要拒答 —— 这比调阈值可靠得多。")
    else:
        print("  该答对的问题都在这条线之上，可以放心使用。")
    if recommended > settings.min_vector_score:
        print(f"\n  当前配置是 {settings.min_vector_score}：在这份语料上**拦不住任何候选**。")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
