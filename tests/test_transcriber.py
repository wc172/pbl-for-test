"""
测试: 离线转录模块
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.transcriber import AudioTranscriber, transcribe_audio


class TestTranscriber(unittest.TestCase):
    """测试转录器"""
    
    def test_01_initialization(self):
        """测试初始化"""
        transcriber = AudioTranscriber("test_course")
        self.assertEqual(transcriber.course_name, "test_course")
    
    def test_02_audio_not_exist(self):
        """测试音频不存在"""
        transcriber = AudioTranscriber("nonexistent")
        with self.assertRaises(FileNotFoundError):
            transcriber.transcribe()
    
    def test_03_transcribe_function(self):
        """测试便捷函数存在"""
        self.assertTrue(callable(transcribe_audio))


def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestTranscriber))
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite).wasSuccessful()


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
