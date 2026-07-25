from __future__ import annotations

import re
import unittest

from main import QWEN_TTS_BRIDGE_SOURCE


class TTSBridgeTests(unittest.TestCase):
    def test_embedded_bridge_compiles_and_splits_without_losing_text(self) -> None:
        namespace = {"__name__": "podcast_radar_tts_bridge_test"}
        exec(compile(QWEN_TTS_BRIDGE_SOURCE, "<qwen3_tts_bridge>", "exec"), namespace)
        split_tts_chunks = namespace["split_tts_chunks"]
        source = "。".join(f"这是第 {index} 段需要朗读的中文内容" for index in range(1, 180))

        chunks = split_tts_chunks(source, 600)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 600 for chunk in chunks))
        self.assertEqual(
            re.sub(r"\s+", "", source),
            re.sub(r"\s+", "", "".join(chunks)),
        )
        self.assertIn(
            'out.with_name(f"{out.stem}.part-{os.getpid()}{out.suffix}")',
            QWEN_TTS_BRIDGE_SOURCE,
        )


if __name__ == "__main__":
    unittest.main()
