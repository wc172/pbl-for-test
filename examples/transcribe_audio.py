"""
示例: 音频转录

使用 FunASR/Paraformer 进行语音识别，生成带时间戳的SRT字幕
"""

import sys
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.transcriber import AudioTranscriber, transcribe_audio


def load_config():
    """加载配置文件"""
    config_path = Path("config/pipeline.yaml")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="音频转录")
    parser.add_argument("--course", required=True, help="课程名称")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--hotword", default="", help="热词列表")
    
    args = parser.parse_args()
    
    # 加载配置
    full_config = load_config()
    trans_config = full_config.get('transcription', {})
    
    # 命令行参数覆盖配置
    if args.device:
        trans_config['device'] = args.device
    if args.hotword:
        trans_config['hotword'] = args.hotword
    
    # 打印模型路径
    model_path = trans_config.get('model', 'paraformer-zh')
    print(f"使用模型: {model_path}")
    
    try:
        print(f"开始转录: {args.course}")
        srt_path = transcribe_audio(args.course, trans_config)
        print(f"转录完成: {srt_path}")
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
