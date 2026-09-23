"""WorldAI 系统异常定义。

所有模块抛出的错误都继承自 :class:`WorldAIError`，便于调用方统一捕获与分类。
"""


class WorldAIError(Exception):
    """系统所有异常的基类。"""


class ToolError(WorldAIError):
    """工具执行失败（如计算器语法错误、日期解析失败）。

    由编排层捕获后作为 Observation 回灌给智能体，而非中断整轮推理。
    """


class ProviderError(WorldAIError):
    """可替换 Provider 加载或执行失败（如模型文件缺失、远端服务不可达）。"""


class IngestError(WorldAIError):
    """文档摄入阶段失败（抽取失败、空文档、分块异常）。"""
