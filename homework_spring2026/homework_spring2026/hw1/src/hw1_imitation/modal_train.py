"""可选的 Modal 云端训练入口；本地完成 HW1 时无需使用本模块。"""

from pathlib import Path

import modal

from hw1_imitation.train import TrainConfig, parse_train_config, run_training


# Modal 应用、持久卷和远端挂载位置的集中配置。
APP_NAME = "hw1-imitation"
NETRC_PATH = Path("~/.netrc").expanduser()
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/vol"
DEFAULT_GPU = "T4"
DEFAULT_CPU = 2.0
volume = modal.Volume.from_name("hw1-imitation-volume", create_if_missing=True)


def load_gitignore_patterns() -> list[str]:
    """把本地 .gitignore 条目转换为 Modal 上传目录时使用的排除 glob。"""

    # 只有构建/上传发生在本地时才需要读取本地 .gitignore。
    if not modal.is_local():
        return []

    root = Path(__file__).resolve().parents[2]
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return []

    patterns: list[str] = []
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("!"):
            continue
        entry = entry.lstrip("/")
        if entry.endswith("/"):
            entry = entry.rstrip("/")
            patterns.append(f"**/{entry}/**")
        else:
            patterns.append(f"**/{entry}")
    return patterns


# 用 uv.lock 构建可复现的 Debian 镜像，并补充渲染环境所需系统动态库。
image = modal.Image.debian_slim().apt_install("libgl1", "libglib2.0-0").uv_sync()
if NETRC_PATH.is_file():
    # 若存在 .netrc，则复制进镜像以便远端访问需要凭据的服务。
    image = image.add_local_file(
        NETRC_PATH,
        remote_path="/root/.netrc",
        copy=True,
    )
# 上传项目源码，同时排除数据、虚拟环境等 .gitignore 内容。
image = image.add_local_dir(
    ".", remote_path=PROJECT_DIR, ignore=load_gitignore_patterns()
)


app = modal.App(APP_NAME)

# 让远端 Python 找到 src 布局的包，并把 WandB 文件写进持久卷。
env = {
    "PYTHONPATH": f"{PROJECT_DIR}/src",
    "WANDB_DIR": f"{VOLUME_PATH}/wandb",
}


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 4,
    env=env,
    image=image,
    gpu=DEFAULT_GPU,
    cpu=DEFAULT_CPU,
)
def train_remote(*args: str) -> None:
    """Modal 远端函数：复用本地训练逻辑，只替换持久化数据目录。"""
    defaults = TrainConfig()
    defaults.data_dir = Path(VOLUME_PATH) / "data"
    config = parse_train_config(
        list(args),
        defaults=defaults,
        description="Train on Modal.",
    )
    run_training(config)
    # 显式提交卷，确保训练日志、数据缓存和 checkpoint 在容器结束后保留。
    volume.commit()
