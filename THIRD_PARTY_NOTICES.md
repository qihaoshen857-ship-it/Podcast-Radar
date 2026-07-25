# 第三方说明

本项目的长音频切分流程参考并整理自 `Qwen3-ASR-Toolkit`：

- 项目地址：https://github.com/QwenLM/Qwen3-ASR-Toolkit
- 许可证：MIT License

整理后的实现位于 `app/transcription.py`，保留了 16 kHz 单声道转换、Silero VAD
切分、固定长度降级、并发调用、顺序合并和重复文本清理思路。当前项目不依赖外部
`Qwen3-ASR-Toolkit` 目录。
