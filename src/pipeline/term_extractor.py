"""
课件术语提取器

从课程材料（markdown/jupyter）自动提取高频技术术语，
并生成可能的ASR误识别映射。
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TermEntry:
    """术语条目"""
    term: str                    # 正确术语
    term_type: str               # 类型: class/function/variable/acronym
    frequency: int               # 出现频次
    possible_errors: List[str]   # 可能的ASR误识别形式


class CourseTermExtractor:
    """
    课程术语提取器
    
    从课件材料自动提取技术术语，并生成语音识别错误映射。
    """
    
    # 英文技术术语模式
    TECH_PATTERNS = [
        # 类名 (CamelCase)
        r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',
        # 全大写缩写 (至少2字母)
        r'\b[A-Z]{2,}\b',
        # snake_case函数/变量名
        r'\b[a-z]+_[a-z_]+\b',
        # 小写缩写+单词 (如openai, chatgpt)
        r'\b[a-z]+[a-z]{2,}\b',
    ]
    
    # 已知的常见错误映射（作为基础）
    BASE_ERRORS = {
        'A3.11_autogen': ['爱do卷', 'adojen', 'ado卷', '奥特真', '奥特根', 'auto真', '爱do jen'],
        'LangChain': ['烂链', '浪链', '浪茜', '烂茜', 'long茜'],
        'A3': ['a三', 'a3', '哎三'],
        'RAG': ['reg', 'REG', 'rag'],
        'Dify': ['define', '低飞', 'dify'],
        'PyTorch': ['皮tor', '拍torch', 'pytorch'],
        'Python': ['爬沈', '派森'],
    }
    
    def __init__(self, course_name: str):
        self.course_name = course_name
        self.materials_dir = Path(f"courses/{course_name}/materials")
        self.extracted_terms: Dict[str, TermEntry] = {}
        
    def extract_from_materials(self) -> Dict[str, TermEntry]:
        """
        从课件材料提取术语
        
        Returns:
            术语字典: {术语: TermEntry}
        """
        if not self.materials_dir.exists():
            logger.warning(f"课件目录不存在: {self.materials_dir}")
            return {}
        
        all_terms = []
        
        # 处理所有课件文件
        for file_path in self.materials_dir.iterdir():
            if file_path.suffix in ['.md', '.ipynb']:
                terms = self._extract_from_file(file_path)
                all_terms.extend(terms)
                logger.info(f"从 {file_path.name} 提取了 {len(terms)} 个术语")
        
        # 统计频次
        term_counts = Counter(all_terms)
        
        # 构建术语条目
        for term, count in term_counts.most_common(100):  # 取前100
            if count >= 2:  # 至少出现2次
                errors = self._generate_error_forms(term)
                term_type = self._classify_term(term)
                
                self.extracted_terms[term] = TermEntry(
                    term=term,
                    term_type=term_type,
                    frequency=count,
                    possible_errors=errors
                )
        
        logger.info(f"共提取 {len(self.extracted_terms)} 个高频术语")
        return self.extracted_terms
    
    def _extract_from_file(self, file_path: Path) -> List[str]:
        """从单个文件提取术语"""
        content = self._read_file(file_path)
        terms = []
        
        # 应用各种模式匹配
        for pattern in self.TECH_PATTERNS:
            matches = re.findall(pattern, content)
            # 过滤掉常见英文单词
            matches = [m for m in matches if not self._is_common_english(m)]
            terms.extend(matches)
        
        # 提取代码中的特定模式（如Agent、Config等）
        code_terms = self._extract_code_terms(content)
        terms.extend(code_terms)
        
        return terms
    
    def _read_file(self, file_path: Path) -> str:
        """读取文件内容"""
        if file_path.suffix == '.ipynb':
            return self._read_notebook(file_path)
        else:
            return file_path.read_text(encoding='utf-8')
    
    def _read_notebook(self, file_path: Path) -> str:
        """读取notebook，提取文本内容"""
        try:
            import nbformat
            nb = nbformat.read(file_path, as_version=4)
            
            texts = []
            for cell in nb.cells:
                if cell.cell_type in ['markdown', 'code']:
                    texts.append(cell.source)
            
            return '\n'.join(texts)
        except ImportError:
            logger.warning("未安装nbformat，跳过notebook解析")
            return ""
    
    def _extract_code_terms(self, content: str) -> List[str]:
        """从代码中提取特定术语"""
        terms = []
        
        # 类实例化 pattern = ClassName(
        class_pattern = r'\b([A-Z][a-zA-Z]+)\s*\('
        terms.extend(re.findall(class_pattern, content))
        
        # 配置项 config_key = value
        config_pattern = r'\b([a-z_]+)\s*=\s*["\']?\w+["\']?'
        terms.extend(re.findall(config_pattern, content))
        
        # 参数名 parameter=True/False
        param_pattern = r'\b([a-z_]+)\s*=\s*(?:True|False)'
        terms.extend(re.findall(param_pattern, content))
        
        return terms
    
    def _is_common_english(self, word: str) -> bool:
        """检查是否为常见英文单词（非术语）"""
        common_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
            'did', 'she', 'use', 'her', 'way', 'many', 'oil', 'sit', 'set', 'run',
            'eat', 'far', 'sea', 'eye', 'ago', 'off', 'too', 'any', 'say', 'man',
            'try', 'ask', 'end', 'why', 'let', 'put', 'say', 'she', 'try', 'way',
            'own', 'say', 'too', 'old', 'tell', 'very', 'when', 'come', 'could',
            'would', 'there', 'their', 'what', 'said', 'each', 'which', 'will',
            'about', 'could', 'other', 'after', 'first', 'never', 'these', 'think',
            'where', 'being', 'every', 'great', 'might', 'shall', 'still', 'those',
            'while', 'this', 'that', 'with', 'have', 'from', 'they', 'been',
            'were', 'said', 'time', 'than', 'them', 'into', 'just', 'like',
            'over', 'also', 'back', 'only', 'know', 'take', 'year', 'good',
            'some', 'come', 'make', 'well', 'work', 'life', 'even', 'here',
            'look', 'down', 'most', 'long', 'last', 'find', 'give', 'does',
            'made', 'part', 'such', 'keep', 'call', 'came', 'need', 'feel',
            'seem', 'turn', 'hand', 'high', 'sure', 'upon', 'head', 'help',
            'home', 'side', 'move', 'both', 'five', 'once', 'same', 'must',
            'name', 'left', 'each', 'done', 'open', 'case', 'show', 'live',
            'play', 'went', 'told', 'seen', 'hear', 'talk', 'soon', 'read',
            'stop', 'face', 'fact', 'land', 'line', 'kind', 'next', 'word',
            'came', 'went', 'told', 'seen', 'hear', 'talk', 'soon', 'read',
            'Self', 'True', 'False', 'None', 'class', 'def', 'return', 'import',
            'print', 'len', 'range', 'list', 'dict', 'str', 'int', 'float',
            'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'finally',
            'with', 'as', 'from', 'pass', 'break', 'continue', 'lambda',
        }
        return word.lower() in common_words
    
    def _classify_term(self, term: str) -> str:
        """分类术语类型"""
        if term[0].isupper() and '_' not in term:
            return 'class'
        elif term.isupper():
            return 'acronym'
        elif '_' in term:
            return 'variable'
        else:
            return 'function'
    
    def _generate_error_forms(self, term: str) -> List[str]:
        """
        生成可能的ASR误识别形式
        
        基于拼音相似性和常见ASR错误模式
        """
        errors = []
        
        # 1. 检查基础错误映射
        for correct, error_list in self.BASE_ERRORS.items():
            if term.lower() == correct.lower():
                errors.extend(error_list)
        
        # 2. 生成拼音相似形式（如果是英文单词）
        if term.isalpha():
            errors.extend(self._phonetic_similar(term))
        
        # 3. 分隔驼峰命名
        if self._is_camel_case(term):
            parts = self._split_camel_case(term)
            # 为每个部分生成错误
            for part in parts:
                if len(part) >= 3:
                    errors.extend(self._phonetic_similar(part))
        
        return list(set(errors))  # 去重
    
    def _phonetic_similar(self, word: str) -> List[str]:
        """生成拼音相似的ASR错误"""
        # 简化的拼音映射（可扩展为完整的拼音库）
        phonetic_map = {
            'au': ['奥', '澳', '傲'],
            'to': ['托', '图', '途'],
            'gen': ['真', '根', '珍'],
            'lang': ['浪', '朗', '郎'],
            'chain': ['茜', '链', '建'],
            'py': ['皮', '派', '批'],
            'torch': ['tor', '投', '拖'],
            'auto': ['奥特', '奥托', '奥'],
            'config': ['康菲', '康菲g'],
            'agent': ['爱真', '爱珍', '埃真t'],
        }
        
        errors = []
        word_lower = word.lower()
        
        for pattern, chars in phonetic_map.items():
            if pattern in word_lower:
                for char in chars:
                    error = word_lower.replace(pattern, char)
                    if error != word_lower:
                        errors.append(error)
        
        return errors
    
    def _is_camel_case(self, s: str) -> bool:
        """检查是否为驼峰命名"""
        return s[0].isupper() and any(c.isupper() for c in s[1:])
    
    def _split_camel_case(self, s: str) -> List[str]:
        """拆分驼峰命名字符串"""
        import re
        matches = re.finditer(r'[A-Z][a-z]*', s)
        return [m.group(0) for m in matches]
    
    def generate_correction_dict(self) -> Dict[str, str]:
        """
        生成纠错字典
        
        Returns:
            {错误形式: 正确术语}
        """
        correction_dict = {}
        
        for term, entry in self.extracted_terms.items():
            for error in entry.possible_errors:
                correction_dict[error] = term
        
        return correction_dict
    
    def save_to_file(self, output_path: Path = None):
        """保存提取的术语到文件"""
        if output_path is None:
            output_path = Path(f"courses/{self.course_name}/extracted_terms.json")
        
        data = {
            term: {
                'type': entry.term_type,
                'frequency': entry.frequency,
                'possible_errors': entry.possible_errors
            }
            for term, entry in self.extracted_terms.items()
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"术语已保存: {output_path}")


def extract_course_terms(course_name: str) -> Dict[str, str]:
    """
    便捷函数：提取课程术语并生成纠错字典
    
    Args:
        course_name: 课程名称
        
    Returns:
        纠错字典: {错误形式: 正确术语}
    
    Example:
        >>> errors = extract_course_terms("A3.11_autogen")
        >>> print(errors.get("爱do卷"))  # "A3.11_autogen"
    """
    extractor = CourseTermExtractor(course_name)
    extractor.extract_from_materials()
    extractor.save_to_file()
    return extractor.generate_correction_dict()


if __name__ == "__main__":
    # 测试
    import sys
    
    if len(sys.argv) > 1:
        course = sys.argv[1]
    else:
        course = "A3.11_autogen"
    
    print(f"提取课程 '{course}' 的术语...")
    
    errors = extract_course_terms(course)
    
    print(f"\n生成了 {len(errors)} 个错误映射:")
    for error, correct in list(errors.items())[:20]:
        print(f'  "{error}" -> "{correct}"')
