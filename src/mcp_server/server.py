"""
模块5: MCP服务器 - FastMCP实现
stdio模式，纯协议适配层，零业务逻辑
"""

import sys
import json
import logging
import asyncio
import signal
from typing import Dict, Any, Optional, List
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# 确保项目根目录在路径中
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 延迟导入底层模块（避免启动时加载重库）
VideoContentStore = None
CourseRAGQueryInterface = None

def _import_video_store():
    global VideoContentStore
    if VideoContentStore is None:
        from src.pipeline.video_store import VideoContentStore as VCS
        VideoContentStore = VCS
    return VideoContentStore

def _import_rag_interface():
    global CourseRAGQueryInterface
    if CourseRAGQueryInterface is None:
        from src.pipeline.course_rag import CourseRAGQueryInterface as CRI
        CourseRAGQueryInterface = CRI
    return CourseRAGQueryInterface

# 配置日志 - 只写入文件，不输出到控制台（避免污染MCP通信）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(project_root / "logs" / "mcp_server.log"), mode='a')
    ]
)
logger = logging.getLogger(__name__)


class TranscriptionMCPServer:
    """MCP服务 - 纯协议适配层"""
    
    def __init__(self):
        self.name = "video-transcription-server"
        self.version = "2.0.0"
        
        # 缓存实例（避免重复创建）
        self._video_stores: Dict[str, Any] = {}
        self._rag_interfaces: Dict[str, Any] = {}
        
        # 创建FastMCP实例
        self.mcp = FastMCP(self.name)
        self._register_tools()
    
    def _get_video_store(self, course_name: str) -> Any:
        """获取或创建VideoContentStore实例"""
        # course_name保持原样，大小写敏感
        if course_name not in self._video_stores:
            VCS = _import_video_store()
            self._video_stores[course_name] = VCS(course_name)
        return self._video_stores[course_name]
    
    def _get_rag_interface(self, course_name: str) -> Any:
        """获取或创建CourseRAGQueryInterface实例"""
        # course_name保持原样，大小写敏感
        if course_name not in self._rag_interfaces:
            CRI = _import_rag_interface()
            self._rag_interfaces[course_name] = CRI(course_name)
        return self._rag_interfaces[course_name]
    
    def _list_available_courses(self) -> Dict[str, Any]:
        """
        扫描所有已处理的课程
        
        检查两个数据源：
        - 视频索引: .cache/{course_name}/facts/segments.jsonl
        - 课件RAG: vector_db/course_materials/{course_name}/chroma.sqlite3
        
        Returns:
            课程列表，每项包含名称和数据就绪状态
        """
        courses = []
        
        # 先导入VideoContentStore（延迟导入）
        VCS = _import_video_store()
        
        # 扫描视频索引（.cache目录）
        cache_dir = project_root / ".cache"
        video_courses = set()
        if cache_dir.exists():
            for item in cache_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    course_name = item.name
                    store = VCS(course_name)
                    if store.exists():
                        video_courses.add(course_name)
        
        # 扫描课件RAG（vector_db目录）
        vector_db_dir = project_root / "vector_db" / "course_materials"
        rag_courses = set()
        if vector_db_dir.exists():
            for item in vector_db_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # 检查是否有chroma.sqlite3文件
                    if (item / "chroma.sqlite3").exists():
                        rag_courses.add(item.name)
        
        # 合并所有课程
        all_courses = video_courses | rag_courses
        
        for course_name in sorted(all_courses):
            courses.append({
                "name": course_name,
                "has_video_index": course_name in video_courses,
                "has_material_rag": course_name in rag_courses
            })
        
        return {"courses": courses}
    
    def _register_tools(self):
        """注册所有MCP工具"""
        
        @self.mcp.tool()
        async def video_get_course_structure(course_name: str) -> List[Dict[str, Any]]:
            """获取视频课程结构导航"""
            try:
                logger.info(f"[video_get_course_structure] course={course_name}")
                store = self._get_video_store(course_name)
                if not store.exists():
                    return [{"error": f"Course not found: {course_name}"}]
                result = store.get_course_structure()
                logger.info(f"  -> 返回 {len(result)} 个segments")
                return result
            except Exception as e:
                logger.error(f"Error: {e}")
                return [{"error": str(e)}]
        
        @self.mcp.tool()
        async def video_get_segment_by_time(course_name: str, timestamp: float) -> Optional[Dict[str, Any]]:
            """根据时间戳定位Segment"""
            try:
                logger.info(f"[video_get_segment_by_time] course={course_name}, timestamp={timestamp}")
                store = self._get_video_store(course_name)
                if not store.exists():
                    return {"error": f"Course not found: {course_name}"}
                return store.get_segment_by_time(timestamp)
            except Exception as e:
                logger.error(f"Error: {e}")
                return {"error": str(e)}
        
        @self.mcp.tool()
        async def video_get_segments_by_concept(course_name: str, concept: str) -> List[Dict[str, Any]]:
            """根据概念关键词定位Segments"""
            try:
                logger.info(f"[video_get_segments_by_concept] course={course_name}, concept={concept}")
                store = self._get_video_store(course_name)
                if not store.exists():
                    return [{"error": f"Course not found: {course_name}"}]
                return store.get_segments_by_concept(concept)
            except Exception as e:
                logger.error(f"Error: {e}")
                return [{"error": str(e)}]
        
        @self.mcp.tool()
        async def video_get_segments_by_type(course_name: str, content_type: str) -> List[Dict[str, Any]]:
            """根据内容类型筛选Segments"""
            try:
                logger.info(f"[video_get_segments_by_type] course={course_name}, type={content_type}")
                store = self._get_video_store(course_name)
                if not store.exists():
                    return [{"error": f"Course not found: {course_name}"}]
                return store.get_segments_by_type(content_type)
            except Exception as e:
                logger.error(f"Error: {e}")
                return [{"error": str(e)}]
        
        @self.mcp.tool()
        async def video_get_segment_content(course_name: str, segment_id: str) -> Optional[Dict[str, Any]]:
            """获取单个Segment完整内容（含原文）"""
            try:
                logger.info(f"[video_get_segment_content] course={course_name}, segment_id={segment_id}")
                store = self._get_video_store(course_name)
                if not store.exists():
                    return {"error": f"Course not found: {course_name}"}
                return store.get_segment_content(segment_id)
            except Exception as e:
                logger.error(f"Error: {e}")
                return {"error": str(e)}
        
        @self.mcp.tool()
        async def video_get_multiple_segments(course_name: str, segment_ids: List[str]) -> List[Dict[str, Any]]:
            """批量获取多个Segments内容"""
            try:
                logger.info(f"[video_get_multiple_segments] course={course_name}, ids={segment_ids}")
                store = self._get_video_store(course_name)
                if not store.exists():
                    return [{"error": f"Course not found: {course_name}"}]
                return store.get_multiple_segments(segment_ids)
            except Exception as e:
                logger.error(f"Error: {e}")
                return [{"error": str(e)}]
        
        @self.mcp.tool()
        async def video_get_raw_text_range(course_name: str, start_sec: float, end_sec: float) -> Dict[str, Any]:
            """获取时间范围内的原始文本拼接"""
            try:
                logger.info(f"[video_get_raw_text_range] course={course_name}, range=[{start_sec}, {end_sec}]")
                store = self._get_video_store(course_name)
                if not store.exists():
                    return {"error": f"Course not found: {course_name}"}
                return store.get_raw_text_range(start_sec, end_sec)
            except Exception as e:
                logger.error(f"Error: {e}")
                return {"error": str(e)}
        
        @self.mcp.tool()
        async def video_get_segment_metadata(course_name: str, segment_id: str) -> Optional[Dict[str, Any]]:
            """获取Segment轻量级元数据（无原文）"""
            try:
                logger.info(f"[video_get_segment_metadata] course={course_name}, segment_id={segment_id}")
                store = self._get_video_store(course_name)
                if not store.exists():
                    return {"error": f"Course not found: {course_name}"}
                return store.get_segment_metadata(segment_id)
            except Exception as e:
                logger.error(f"Error: {e}")
                return {"error": str(e)}
        
        @self.mcp.tool()
        async def video_get_course_stats(course_name: str) -> Dict[str, Any]:
            """获取课程统计信息"""
            try:
                logger.info(f"[video_get_course_stats] course={course_name}")
                store = self._get_video_store(course_name)
                if not store.exists():
                    return {"error": f"Course not found: {course_name}"}
                return store.get_course_stats()
            except Exception as e:
                logger.error(f"Error: {e}")
                return {"error": str(e)}
        
        @self.mcp.tool()
        async def video_get_navigation_map(course_name: str) -> Dict[str, Any]:
            """获取全局导航图"""
            try:
                logger.info(f"[video_get_navigation_map] course={course_name}")
                store = self._get_video_store(course_name)
                if not store.exists():
                    return {"error": f"Course not found: {course_name}"}
                return store.get_navigation_map()
            except Exception as e:
                logger.error(f"Error: {e}")
                return {"error": str(e)}
        
        @self.mcp.tool()
        async def material_search(course_name: str, query: str, top_k: int = 3) -> Dict[str, Any]:
            """语义搜索课件内容"""
            try:
                logger.info(f"[material_search] course={course_name}, query='{query}', top_k={top_k}")
                rag = self._get_rag_interface(course_name)
                if not rag.exists():
                    return {"error": f"Course RAG not found: {course_name}"}
                
                results = rag.search(query, top_k=top_k)
                formatted_results = [
                    {
                        "id": r["id"],
                        "text": r["text"],
                        "source_file": r["metadata"].get("source_file", ""),
                        "headings": json.loads(r["metadata"].get("headings", "[]")),
                        "similarity": r.get("similarity", 1 - r.get("distance", 0))
                    }
                    for r in results
                ]
                return {
                    "query": query,
                    "results": formatted_results
                }
            except Exception as e:
                logger.error(f"Error: {e}")
                return {"error": str(e)}
        
        @self.mcp.tool()
        async def material_batch_search(course_name: str, queries: List[str], top_k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
            """批量语义搜索课件"""
            try:
                logger.info(f"[material_batch_search] course={course_name}, queries={queries}")
                rag = self._get_rag_interface(course_name)
                if not rag.exists():
                    return {"error": f"Course RAG not found: {course_name}"}
                
                batch_results = rag.batch_search(queries, top_k=top_k)
                return batch_results
            except Exception as e:
                logger.error(f"Error: {e}")
                return {"error": str(e)}
        
        @self.mcp.tool()
        async def list_available_courses() -> Dict[str, Any]:
            """列出所有可用的课程"""
            try:
                logger.info("[list_available_courses]")
                result = self._list_available_courses()
                logger.info(f"  -> 返回 {len(result['courses'])} 个课程")
                return result
            except Exception as e:
                logger.error(f"Error: {e}")
                return {"error": str(e), "courses": []}
    
    def run_stdio(self):
        """以stdio模式运行"""
        def signal_handler(signum, frame):
            """处理终止信号"""
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            sys.exit(0)
        
        # 注册信号处理
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
        
        logger.info(f"Starting {self.name} v{self.version} in stdio mode...")
        try:
            self.mcp.run(transport='stdio')
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, shutting down...")
        except asyncio.CancelledError:
            logger.info("CancelledError received, shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
        finally:
            logger.info("Server shutdown complete")


_server: Optional[TranscriptionMCPServer] = None


def get_server() -> TranscriptionMCPServer:
    """获取或创建服务器实例（单例模式）"""
    global _server
    if _server is None:
        _server = TranscriptionMCPServer()
    return _server


def main():
    """主入口"""
    try:
        server = get_server()
        server.run_stdio()
    except KeyboardInterrupt:
        print("\nServer stopped by user (Ctrl+C)", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
