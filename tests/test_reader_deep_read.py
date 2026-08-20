from app.research_digest import DigestMetadata, build_digest_prompt
from main import normalize_research_digest_markdown, parse_reader_markdown


def test_digest_prompt_preserves_all_core_points_without_value_tier() -> None:
    prompt = build_digest_prompt(
        "A complete transcript with facts, examples, and disagreements.",
        DigestMetadata(title="Deep research episode"),
    )

    assert "## 全文主线" in prompt
    assert "不设条目数量上限" in prompt
    assert "不得因内容较多而删减、合并" in prompt
    assert "## 关键数据与案例" in prompt
    assert "数据与案例台账" in prompt
    assert "所有数字及其上下文" in prompt
    assert "所有定性但可核查的经营或产业信号" in prompt
    assert "不得只挑代表性数据" in prompt
    assert "不同口径、时期、主体或限定条件不得合并" in prompt
    assert "每个带有事实、数字、主体动作、案例或定性产业信号的条目" in prompt
    assert "不要输出内部台账式英文速记" in prompt
    assert "## 后续追踪" in prompt
    assert "不得超过 3 项" in prompt
    assert "## 与我的研究相关" not in prompt
    assert "是否值得我继续深听" not in prompt


def test_legacy_digest_is_normalized_without_cutting_core_points() -> None:
    legacy = """# 测试纪要

## 一句话判断
**值得深听**：这是需要完整理解的核心主线。

## 核心要点
1. 要点一
2. 要点二
3. 要点三
4. 要点四
5. 要点五

## 与我的研究相关
这个栏目应该隐藏。

## 可继续跟踪的问题
1. 问题一
   - 观察信号一
2. 问题二
3. 问题三
4. 问题四
5. 问题五

## 关键词
AI、Agent
"""

    normalized = normalize_research_digest_markdown(legacy)

    assert "## 全文主线" in normalized
    assert "值得深听" not in normalized
    assert "这是需要完整理解的核心主线" in normalized
    assert "与我的研究相关" not in normalized
    assert "要点五" in normalized
    assert "## 后续追踪" in normalized
    assert "问题三" in normalized
    assert "观察信号一" in normalized
    assert "问题四" not in normalized
    assert "问题五" not in normalized
    assert "## 关键词" in normalized


def test_markdown_parser_joins_wrapped_list_lines_and_keeps_nested_depth() -> None:
    markdown = """## 核心要点
1. **企业 AI**：第一行说明主结论，
   第二行继续补充证据和条件。
   - 子项观察信号
2. **产品化**：另一个要点。

这是一个很长段落的第一行，
第二行应该与上一行合并成同一阅读段落。
"""

    blocks = parse_reader_markdown(markdown)
    content_blocks = [block for block in blocks if block.kind != "spacer"]

    assert content_blocks[0].kind == "h2"
    assert content_blocks[1].kind == "list"
    assert content_blocks[1].marker == "1."
    assert "第一行说明" in content_blocks[1].text
    assert "第二行继续补充" in content_blocks[1].text
    assert content_blocks[2].kind == "list"
    assert content_blocks[2].depth == 1
    assert content_blocks[2].text == "子项观察信号"
    assert content_blocks[3].kind == "list"
    assert content_blocks[3].marker == "2."
    assert content_blocks[4].kind == "body"
    assert "第二行应该与上一行合并" in content_blocks[4].text
