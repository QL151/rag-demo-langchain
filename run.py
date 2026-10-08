"""开发入口（给 PyCharm 和命令行用）。

用法：
    python run.py                      # 默认 8000 端口
    python run.py 8001                 # 换端口
    python run.py --reload             # 改代码自动重启（热重载）
    python run.py 8001 --reload        # 两个参数可以任意顺序组合

在 PyCharm 里：右键本文件 → Run 'run'，或点行号旁的绿色 ▶。
（想让它也热重载，就在 Run Configuration 的 Script parameters 里填 --reload）

为什么需要这个文件？
    app/main.py 用的是包内相对导入（from .config import ...），
    直接在 PyCharm 里"运行 app/main.py"会因为它不是包的一部分而报
    "attempted relative import with no known parent package"。
    放在项目根目录的 run.py 没有这个问题。

注意：--reload 需要 watchfiles（pip install watchfiles，装 chromadb 时会顺带装上）。
      它只监听 .py 文件的改动；改了 .env 仍需手动重启。
"""

from __future__ import annotations

import sys

import uvicorn

if __name__ == "__main__":
    # 把所有非 -- 开头的参数当作端口号（这样 --reload 放前面后面都行）
    positional = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    port = int(positional[0]) if positional else 8000
    reload_enabled = "--reload" in sys.argv

    uvicorn.run("app.main:app", host="127.0.0.1", port=port, reload=reload_enabled)
