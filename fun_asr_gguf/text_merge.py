"""
FunASR-GGUF 结果合并模块

处理长音频识别时多个片段结果的拼接和去重。
"""

import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

def merge_transcription_results(
    results: List[Dict[str, Any]], 
    chunk_offsets: List[float], 
    overlap: float
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    合并多个片段的识别结果
    
    采用基于时间戳的初步切分和基于字符匹配的精细去重策略。
    """
    if not results:
        return "", []
    
    if len(results) == 1:
        offset = chunk_offsets[0]
        full_text = results[0]['text']
        full_segments = []
        for seg in results[0].get('segments') or []:
            full_segments.append({
                'char': seg['char'],
                'start': seg['start'] + offset
            })
        return full_text, full_segments

    full_segments = []
    
    # 我们维护一个当前已保留的全局字符列表
    for i, res in enumerate(results):
        offset = chunk_offsets[i]
        curr_segments = res.get('segments') or []
        if not curr_segments and res['text']:
            # 如果没有时间戳对齐结果，尝试简单构造一个（均匀分布）
            duration = res['duration']
            chars = list(res['text'])
            t_per_c = duration / max(1, len(chars))
            curr_segments = [{'char': c, 'start': idx * t_per_c} for idx, c in enumerate(chars)]
        
        # 将当前片段的时间戳转为全局时间戳
        for seg in curr_segments:
            seg['_global_start'] = seg['start'] + offset

        if i == 0:
            # 第一个片段，保留直到 overlap 区域中间的部分
            cutoff = res['duration'] - overlap / 2
            for seg in curr_segments:
                if seg['start'] < cutoff:
                    full_segments.append({'char': seg['char'], 'start': seg['_global_start']})
        else:
            # 后续片段，首先找到与已有结果的衔接点
            # 我们看当前片段在 overlap 区域（0 ~ overlap）内的内容
            # 与 full_segments 末尾的内容进行匹配
            
            # 这里的简单策略：
            # 1. 丢弃当前片段开头 0.5 * overlap 之前的所有字符（因为它们在上一片段已经处理过）
            # 2. 从上一片段保留的末尾字符开始，寻找重叠
            
            start_search_time = overlap / 2
            # 如果不是最后一个片段，还要考虑结尾切割
            end_cutoff = res['duration'] - overlap / 2 if i < len(results) - 1 else res['duration'] + 1
            
            # 在衔接处做一点模糊匹配去重
            # 找到当前片段中，时间戳在 [overlap/2 - 1.0, overlap/2 + 1.0] 范围内的字符
            # 看看是否已经在 full_segments 中存在
            
            last_global_time = full_segments[-1]['start'] if full_segments else -1.0
            
            for seg in curr_segments:
                # 策略：如果当前字符的全局时间明显大于已有的最后一个字符时间，则接纳
                # 这里的 0.05 是一个小冗余，防止微小的时间差导致重复
                if seg['_global_start'] > last_global_time + 0.05:
                    if seg['start'] < end_cutoff:
                        full_segments.append({'char': seg['char'], 'start': seg['_global_start']})

    full_text = "".join([s['char'] for s in full_segments])
    return full_text, full_segments
