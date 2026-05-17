"""
断点续传工具模块

提供统一的断点保存和恢复功能。
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class CheckpointManager:
    """断点管理器"""
    
    def __init__(self, course_name: str):
        self.course_name = course_name
        self.checkpoint_dir = Path(f".cache/{course_name}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, step_name: str, data: Dict[str, Any]):
        """保存断点"""
        checkpoint_file = self.checkpoint_dir / f"{step_name}.checkpoint"
        checkpoint_data = {
            "course_name": self.course_name,
            "step_name": step_name,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    
    def load(self, step_name: str) -> Optional[Dict[str, Any]]:
        """加载断点"""
        checkpoint_file = self.checkpoint_dir / f"{step_name}.checkpoint"
        if not checkpoint_file.exists():
            return None
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def exists(self, step_name: str) -> bool:
        """检查断点是否存在"""
        checkpoint_file = self.checkpoint_dir / f"{step_name}.checkpoint"
        return checkpoint_file.exists()
    
    def delete(self, step_name: str):
        """删除断点"""
        checkpoint_file = self.checkpoint_dir / f"{step_name}.checkpoint"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
    
    def get_progress(self, step_name: str) -> float:
        """获取进度百分比"""
        checkpoint = self.load(step_name)
        if not checkpoint:
            return 0.0
        data = checkpoint.get("data", {})
        processed = data.get("processed_seconds", 0)
        total = data.get("total_duration", 1)
        return min(100.0, (processed / total) * 100) if total > 0 else 0.0


def compute_file_hash(file_path: str) -> str:
    """计算文件MD5哈希"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
