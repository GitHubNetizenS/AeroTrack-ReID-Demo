"""
run_demo.py文件
    该文件是最终Demo总入口，调用demo_pipeline中的完整流水线。
"""

from demo_pipeline import run_demo


if __name__ == "__main__":
    run_demo("configs/demo_config.yaml")
