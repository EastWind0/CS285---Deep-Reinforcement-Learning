"""把 HW2 训练提交到 Modal 云端的入口。

该脚本复用 ``scripts.run`` 的参数解析和训练逻辑，只额外描述云端镜像、资源、
持久化卷和待上传的项目文件。作业规模较小，官方更推荐本地 CPU；此入口主要是
本机依赖安装困难或希望后台运行时的备选方案。
"""

from pathlib import Path

import modal

from scripts.run import main, setup_arguments


# Modal 应用名与云端运行目录。
APP_NAME = "hw2-pg"
NETRC_PATH = Path("~/.netrc").expanduser()
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/root/exp"
# 默认申请的计算资源；可根据账号配额修改。HW2 通常不需要 GPU。
DEFAULT_GPU = "T4"
DEFAULT_CPU = 2.0
DEFAULT_MEMORY = 4096  # MB
# exp 目录挂载为持久化卷，使容器退出后日志仍然存在。
volume = modal.Volume.from_name("hw2-pg-volume", create_if_missing=True)


def load_gitignore_patterns() -> list[str]:
    """把本地 .gitignore 条目转换成 Modal 上传时使用的 glob。"""

    # 云端构建阶段不需要读取本机文件系统；只在本地组装镜像时处理。
    if not modal.is_local():
        return []

    root = Path(__file__).resolve().parents[2]
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return []

    patterns: list[str] = []
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        # 跳过空行、注释和 gitignore 的“重新包含”规则；这里只构造排除列表。
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("!"):
            continue
        entry = entry.lstrip("/")
        if entry.endswith("/"):
            # 目录规则扩展为递归 glob，排除目录下全部内容。
            entry = entry.rstrip("/")
            patterns.append(f"**/{entry}/**")
        else:
            patterns.append(f"**/{entry}")
    return patterns


# 使用 Debian slim 创建镜像；OpenGL 库支持 MuJoCo/渲染，SWIG 用于 Linux 下
# 构建 box2d-py，uv_sync 会依据 pyproject.toml 和 uv.lock 安装其余依赖。
image = modal.Image.debian_slim().apt_install("libgl1", "libglib2.0-0", "swig").uv_sync()
# 若本机已有 .netrc，则复制进镜像，让远端训练可直接认证 W&B。
if NETRC_PATH.is_file():
    image = image.add_local_file(
        NETRC_PATH,
        remote_path="/root/.netrc",
        copy=True,
    )
# 上传项目源码，同时遵守 .gitignore，避免上传 .venv、日志等大文件。
image = image.add_local_dir(
    ".", remote_path=PROJECT_DIR, ignore=load_gitignore_patterns()
)


# Modal 应用对象负责注册下面的远程函数。
app = modal.App(APP_NAME)

# 让远端 Python 能按 ``from agents...`` 等顶层包路径导入 src 中模块。
env = {
    "PYTHONPATH": f"{PROJECT_DIR}/src",
}


@app.function(volumes={VOLUME_PATH: volume}, timeout=60 * 60 * 1, env=env, image=image, gpu=DEFAULT_GPU, cpu=DEFAULT_CPU, memory=DEFAULT_MEMORY)
def hw2_modal_remote(*args: str) -> None:
    """解析透传的命令行参数，在云端运行训练并提交持久化卷。"""
    args = setup_arguments(args)
    main(args)
    # 显式提交卷的最新状态，确保训练生成的 exp 日志持久化。
    volume.commit()
