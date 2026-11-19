"""全局配置：模型接入、护栏阈值、路径等可调参数集中管理。

设计说明
--------
- LLM 客户端通过环境变量接入，支持 OpenAI 兼容协议（DeepSeek / 通义 DashScope /
  Qwen / 本地 Ollama 等均可）；未配置 API Key 时自动回退到内置规则引擎，保证项目
  在无网络、无密钥环境下也能跑通 Benchmark 与测试。
- 所有路径相对项目根目录解析，避免在不同机器上出现绝对路径问题。
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    """从项目根目录的 ``.env`` 读取键值对写入环境变量（不覆盖已存在的系统变量）。

    纯标准库实现，避免额外依赖；格式与常见 .env 一致：``KEY=VALUE``，
    支持 ``#`` 注释和首尾引号。
    """
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BENCHMARK_DIR = DATA_DIR / "benchmark"
DOCS_DIR = BASE_DIR / "docs"

# 先加载 .env，再读取环境变量
_load_dotenv()

# ---------------------------------------------------------------------------
# LLM 接入（OpenAI 兼容协议）
# ---------------------------------------------------------------------------
# 优先读取环境变量；未设置时使用内置 RuleLLM 作为确定性回退。
LLM_PROVIDER = os.getenv("METRIC_PILOT_LLM", "auto")  # auto | openai | rule
LLM_MODEL = os.getenv("METRIC_PILOT_MODEL", "qwen-plus")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
# 兼容 DashScope 专用变量名；优先 OPENAI_API_KEY
LLM_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
LLM_TEMPERATURE = float(os.getenv("METRIC_PILOT_TEMPERATURE", "0.0"))
LLM_TIMEOUT = int(os.getenv("METRIC_PILOT_TIMEOUT", "30"))
LLM_MAX_RETRIES = int(os.getenv("METRIC_PILOT_MAX_RETRIES", "2"))

# ---------------------------------------------------------------------------
# Agent 运行参数
# ---------------------------------------------------------------------------
MAX_AGENT_STEPS = int(os.getenv("METRIC_PILOT_MAX_STEPS", "6"))
TOOL_TIMEOUT_SECONDS = 10
EVIDENCE_MIN_SAMPLE = 5          # 证据验证要求的最小样本量
ANOMALY_Z_THRESHOLD = 2.0        # 指标偏离基线的 Z 分数阈值
RETRY_ON_TOOL_FAIL = 1           # 工具失败后的重试次数

# ---------------------------------------------------------------------------
# 数据生成 / 业务口径
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
METRIC_TABLE = "metric_daily"
DIMENSION_TABLE = "metric_dimension"
ALLOWED_TABLES = {METRIC_TABLE, DIMENSION_TABLE}
