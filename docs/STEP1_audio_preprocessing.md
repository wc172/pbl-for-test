# STEP 1: 输入预处理模块

> **模块1**: 将输入视频/音频转换为标准格式音频（WAV格式，适配Paraformer），删除原文件节省空间，创建断点标记。
> 
> 依赖：步骤0（课件RAG）

---

## 输入

```yaml
input:
  path: "courses/{course_name}/input.{mp4,mp3,wav,m4a}"
  check: "文件存在性检查"
```

---

## 处理流程

```yaml
pipeline:
  step_1_format_detect:
    description: "检测文件格式及音频编码信息"
    tool: "ffprobe"
    output:
      type: "video" | "audio" | "unsupported"
      codec: str           # 音频编码
      sample_rate: int     # 当前采样率
      need_resample: bool  # 是否需要重采样
    
  step_2_extract_audio:
    description: "统一提取/转换为WAV格式（16kHz, 16bit PCM, 单声道）"
    tool: "ffmpeg"
    
    # 视频提取音频
    video_command: |
      ffmpeg -hide_banner -loglevel error \
        -i {input} \
        -vn -ar 16000 -ac 1 -acodec pcm_s16le \
        -f wav \
        {output}
    
    # 音频转码（MP3/M4A/WAV统一转为标准WAV）
    audio_command: |
      ffmpeg -hide_banner -loglevel error \
        -i {input} \
        -ar 16000 -ac 1 -acodec pcm_s16le \
        -f wav \
        {output}
    
    output: "courses/{course_name}/audio.wav"
    
  step_3_cleanup:
    description: "删除原始文件节省空间"
    action: "删除 input.{mp4,mp3,...}"
    condition: "config.delete_original == true"
    
  step_4_checkpoint:
    description: "创建断点标记，标记步骤完成"
    file: ".cache/{course_name}/step_01_audio.done"
```

---

## 输出

```yaml
output:
  audio: "courses/{course_name}/audio.wav"
    format: "wav"
    codec: "pcm_s16le"   # 16bit PCM
    samplerate: 16000    # Paraformer要求16kHz
    channels: 1
  checkpoint: ".cache/{course_name}/step_01_audio.done"
```

---

## 核心类设计

```python
# src/pipeline/preprocessor.py

class AudioPreprocessor:
    """音频预处理器 - 仅需实例化调用 process()"""
    
    def __init__(self, course_name: str, config: Dict = None)
    def process(self, input_path: str) -> Dict     # 执行完整流程
    
    @staticmethod
    def get_output_path(course_name: str) -> str   # 获取音频路径

# 便捷函数
def preprocess_audio(course_name: str, input_path: str, config: Dict = None) -> Dict
def is_preprocessed(course_name: str) -> bool      # 检查是否已完成
```

---

## 快速开始

### 1. 准备输入文件

将视频/音频放入课程目录：
```bash
cp video.mp4 courses/my_course/
# 或
cp audio.mp3 courses/my_course/
```

### 2. Python 调用

```python
from src.pipeline.preprocessor import AudioPreprocessor

# 初始化预处理器
preprocessor = AudioPreprocessor("my_course")

# 执行预处理（自动检测输入文件）
result = preprocessor.process()
# 返回: {"audio_path": "courses/my_course/audio.wav", ...}
```

### 3. 命令行使用

```bash
# 执行步骤1（预处理）
python -m src.pipeline --course my_course --step 1
```

---

## 下游模块使用

模块2（离线转录）调用示例：

```python
from src.pipeline.preprocessor import AudioPreprocessor, is_preprocessed

# 方式1: 直接检查文件是否存在
if not is_preprocessed("my_course"):
    raise Exception("请先执行步骤1音频预处理")

# 方式2: 直接获取音频路径（无需实例化）
audio_path = AudioPreprocessor.get_output_path("my_course")
# 返回: "courses/my_course/audio.wav"

# 方式3: 使用 Path 直接构造路径（最简单）
from pathlib import Path
audio_path = Path(f"courses/{course_name}/audio.wav")
```

---

## 设计原则

- 模块1**不暴露复杂接口**，仅输出文件
- 其他模块通过**固定路径约定**获取音频文件
- 断点检查通过独立函数 `is_preprocessed()` 完成

---

## 配置项

```yaml
# config/pipeline.yaml

audio:
  ffmpeg_path: "ffmpeg/bin/ffmpeg.exe"
  ffprobe_path: "ffmpeg/bin/ffprobe.exe"
  samplerate: 16000      # Paraformer要求16kHz
  channels: 1            # 单声道
  bit_depth: 16          # 16bit PCM
  delete_original: true  # 是否删除原始文件
```

---

## 文件结构

### 源码文件

```
src/pipeline/preprocessor.py        # 主实现文件
├── AudioPreprocessor              # 音频预处理器类
│   ├── __init__(course_name, config)
│   ├── process(input_path)        # 执行完整预处理流程
│   ├── _detect_format()           # 检测文件格式
│   ├── _extract_audio()           # 提取/转换音频
│   └── _create_checkpoint()       # 创建断点标记
│
├── AudioFormatInfo                # 音频格式信息数据类
│   ├── file_type                  # 'video' | 'audio' | 'unsupported'
│   ├── codec                      # 音频编码
│   ├── sample_rate                # 当前采样率
│   ├── channels                   # 声道数
│   └── need_resample              # 是否需要重采样
│
└── 便捷函数
    ├── preprocess_audio()         # 便捷函数：执行预处理
    └── is_preprocessed()          # 检查是否已完成
```

### 输出文件

```
courses/{course_name}/
└── audio.wav                      # 标准格式音频输出
                                   # 16kHz, 16bit PCM, 单声道

.cache/{course_name}/
└── step_01_audio.done             # 断点标记文件
    # {"status": "completed", "timestamp": "...", "input_hash": "..."}
```

## 故障排除

### 输入文件不存在
```
FileNotFoundError: No input file found
```
**解决:** 确保课程目录下有视频/音频文件（支持 mp4, mp3, wav, m4a）

### FFmpeg 未找到
```
Error: ffmpeg not found
```
**解决:** 配置正确的 FFmpeg 路径
