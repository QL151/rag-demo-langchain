"""打印实际编译图，供学习与核对；在项目根目录执行python -m scripts.show_workflow。"""
from app.main import rag

if __name__ == '__main__':
    print(rag.workflow.get_graph().draw_mermaid())
    print('普通与流式入口都运行编译图；历史存SQLite，当前未使用checkpointer。')
