"""
测试: 音频预处理模块

测试 AudioPreprocessor 的各项功能
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.preprocessor import (
    AudioPreprocessor, 
    AudioFormatInfo, 
    preprocess_audio,
    is_preprocessed
)


class TestAudioPreprocessor(unittest.TestCase):
    """测试音频预处理器"""
    
    @classmethod
    def setUpClass(cls):
        """测试前准备"""
        cls.test_course = "test_preprocessor"
        cls.test_dir = Path(f"courses/{cls.test_course}")
        cls.cache_dir = Path(f".cache/{cls.test_course}")
        
        # 清理之前的测试数据
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
        if cls.cache_dir.exists():
            shutil.rmtree(cls.cache_dir)
    
    @classmethod
    def tearDownClass(cls):
        """测试后清理"""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
        if cls.cache_dir.exists():
            shutil.rmtree(cls.cache_dir)
    
    def setUp(self):
        """每个测试前准备"""
        self.config = {
            "ffmpeg_path": "ffmpeg/bin/ffmpeg.exe",
            "ffprobe_path": "ffmpeg/bin/ffprobe.exe",
            "delete_original": False
        }
        self.preprocessor = AudioPreprocessor(self.test_course, self.config)
    
    def test_01_initialization(self):
        """测试初始化"""
        self.assertEqual(self.preprocessor.course_name, self.test_course)
        self.assertEqual(self.preprocessor.output_path, Path(f"courses/{self.test_course}/audio.wav"))
        self.assertTrue(self.preprocessor.course_dir.exists())
    
    def test_02_detect_format_unsupported(self):
        """测试不支持的格式检测"""
        # 创建一个临时文件
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"test")
            temp_path = f.name
        
        try:
            format_info = self.preprocessor.detect_format(temp_path)
            self.assertEqual(format_info.file_type, "unsupported")
        finally:
            os.unlink(temp_path)
    
    def test_03_detect_format_video(self):
        """测试视频格式检测"""
        # 仅测试扩展名识别（不实际调用 ffprobe）
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(b"fake video data")
            temp_path = f.name
        
        try:
            format_info = self.preprocessor.detect_format(temp_path)
            self.assertEqual(format_info.file_type, "video")
        finally:
            os.unlink(temp_path)
    
    def test_04_detect_format_audio(self):
        """测试音频格式检测"""
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(b"fake audio data")
            temp_path = f.name
        
        try:
            format_info = self.preprocessor.detect_format(temp_path)
            self.assertEqual(format_info.file_type, "audio")
        finally:
            os.unlink(temp_path)
    
    def test_05_audio_format_info_dataclass(self):
        """测试 AudioFormatInfo 数据类"""
        info = AudioFormatInfo(
            file_type="video",
            codec="aac",
            sample_rate=44100,
            channels=2,
            duration=120.5,
            need_resample=True
        )
        
        self.assertEqual(info.file_type, "video")
        self.assertEqual(info.codec, "aac")
        self.assertEqual(info.sample_rate, 44100)
        self.assertTrue(info.need_resample)
    
    def test_06_get_output_path(self):
        """测试获取输出路径"""
        # 测试实例方法（已废弃，改为静态方法）
        output_path = AudioPreprocessor.get_output_path(self.test_course)
        # 使用 Path 比较，自动处理 Windows/Unix 路径分隔符差异
        expected = Path(f"courses/{self.test_course}/audio.wav")
        self.assertEqual(Path(output_path), expected)
    
    def test_07_static_get_output_path(self):
        """测试静态方法获取输出路径"""
        output_path = AudioPreprocessor.get_output_path("my_course")
        self.assertEqual(output_path, "courses/my_course/audio.wav")
    
    def test_08_supported_formats(self):
        """测试支持的格式集合"""
        # 视频格式
        self.assertIn(".mp4", self.preprocessor.SUPPORTED_VIDEO)
        self.assertIn(".avi", self.preprocessor.SUPPORTED_VIDEO)
        
        # 音频格式
        self.assertIn(".wav", self.preprocessor.SUPPORTED_AUDIO)
        self.assertIn(".mp3", self.preprocessor.SUPPORTED_AUDIO)
        self.assertIn(".m4a", self.preprocessor.SUPPORTED_AUDIO)
    
    def test_09_target_params(self):
        """测试目标参数"""
        self.assertEqual(self.preprocessor.TARGET_SAMPLE_RATE, 16000)
        self.assertEqual(self.preprocessor.TARGET_CHANNELS, 1)
        self.assertEqual(self.preprocessor.TARGET_BIT_DEPTH, 16)
        self.assertEqual(self.preprocessor.TARGET_CODEC, "pcm_s16le")


class TestStaticFunctions(unittest.TestCase):
    """测试静态函数"""
    
    def test_is_preprocessed_false(self):
        """测试预处理检查（未处理）"""
        result = is_preprocessed("nonexistent_course")
        self.assertFalse(result)
    
    def test_preprocess_audio_function(self):
        """测试便捷预处理函数（仅测试函数存在）"""
        # 由于需要实际的音频文件，这里只测试函数可被调用
        # 实际调用会在集成测试中完成
        self.assertTrue(callable(preprocess_audio))


class TestAudioFormatInfo(unittest.TestCase):
    """测试 AudioFormatInfo 数据类"""
    
    def test_creation(self):
        """测试创建实例"""
        info = AudioFormatInfo(
            file_type="audio",
            codec="pcm_s16le",
            sample_rate=16000,
            channels=1,
            duration=3600.0,
            need_resample=False
        )
        
        self.assertEqual(info.file_type, "audio")
        self.assertEqual(info.sample_rate, 16000)
        self.assertFalse(info.need_resample)


def run_tests():
    """运行测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestAudioPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioFormatInfo))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
