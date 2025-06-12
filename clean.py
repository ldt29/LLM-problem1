#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikipedia 中文数据清洗工具 - 专为大模型预训练优化

用法:
    python clean.py [dump_file] --output output.jsonl --sample sample.jsonl
"""

import bz2
import re
import json
import argparse
import sys
from pathlib import Path
from html.entities import name2codepoint

# 全局配置
acceptedNamespaces = set(['w'])  # 只接受主命名空间

# 需要丢弃的HTML元素
discardElements = set([
    'gallery', 'timeline', 'noinclude', 'pre', 'table', 'tr', 'td', 'th', 'caption',
    'form', 'input', 'select', 'option', 'textarea', 'ul', 'li', 'ol', 'dl', 'dt', 'dd',
    'menu', 'dir', 'ref', 'references', 'img', 'imagemap', 'source', 'math', 'code'
])

# 忽略的标签（保留内容，去掉标签）
ignoredTags = [
    'b', 'big', 'blockquote', 'center', 'cite', 'div', 'em', 'font', 
    'h1', 'h2', 'h3', 'h4', 'hiero', 'i', 'kbd', 'nowiki', 'p', 
    'plaintext', 's', 'small', 'span', 'strike', 'strong', 'sub', 'sup', 'tt', 'u', 'var'
]

# 自闭合标签
selfClosingTags = ['br', 'hr', 'nobr', 'ref', 'references']

def unescape(text):
    """移除HTML字符引用和实体"""
    def fixup(m):
        text = m.group(0)
        code = m.group(1)
        try:
            if text[1] == "#":  # 字符引用
                if text[2] == "x":
                    return chr(int(code[1:], 16))
                else:
                    return chr(int(code))
            else:  # 命名实体
                return chr(name2codepoint[code])
        except:
            return text  # 保持原样
    return re.sub("&#?(\w+);", fixup, text)

def dropNested(text, openDelim, closeDelim):
    """移除嵌套的结构，如模板{{}}和表格{||}"""
    openRE = re.compile(openDelim)
    closeRE = re.compile(closeDelim)
    matches = []
    nest = 0
    start = openRE.search(text, 0)
    if not start:
        return text
    end = closeRE.search(text, start.end())
    next = start
    while end:
        next = openRE.search(text, next.end())
        if not next:
            while nest:
                nest -= 1
                end0 = closeRE.search(text, end.end())
                if end0:
                    end = end0
                else:
                    break
            matches.append((start.start(), end.end()))
            break
        while end.end() < next.start():
            if nest:
                nest -= 1
                last = end.end()
                end = closeRE.search(text, end.end())
                if not end:
                    if matches:
                        span = (matches[0][0], last)
                    else:
                        span = (start.start(), last)
                    matches = [span]
                    break
            else:
                matches.append((start.start(), end.end()))
                start = next
                end = closeRE.search(text, next.end())
                break
        if next != start:
            nest += 1
    
    # 收集匹配区域外的文本
    res = ''
    start = 0
    for s, e in matches:
        res += text[start:s]
        start = e
    res += text[start:]
    return res

def clean_text(text: str) -> str:
    """
    彻底清洗维基百科文本，专为大模型预训练优化
    """
    
    # ========== 第一阶段：结构性清理 ==========
    
    # 1. 早期截断：移除参考文献等章节后的所有内容
    end_sections = r'(参见|注释|参考文献?|参考书目|外部链接|延伸阅读|相关条目|另见|参考资料|脚注)'
    match = re.search(end_sections, text, re.IGNORECASE)
    if match:
        text = text[:match.start()]
    
    # 2. 移除HTML注释
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    text = dropNested(text, r'{{', r'}}')  # 模板
    text = dropNested(text, r'{\|', r'\|}')  # 表格
    
    # ========== 第二阶段：链接处理 ==========
    
    # 4. 处理维基链接 [[...]]
    # 先处理带显示文本的链接 [[目标|显示文本]]
    def process_wikilink(match):
        link = match.group(1)
        anchor = match.group(2) if match.group(2) else link
        trail = match.group(3) if match.group(3) else ''
        
        # 检查命名空间
        colon = link.find(':')
        if colon > 0:
            namespace = link[:colon].lower()
            if namespace in ['file', 'image', 'media', 'category']:
                return ''  # 完全移除文件/分类链接
        
        return anchor + trail
    
    # 匹配维基链接
    wikiLink = re.compile(r'\[\[([^[]*?)(?:\|([^[]*?))?\]\](\w*)')
    text = wikiLink.sub(process_wikilink, text)
    
    # 移除剩余的参数化链接
    text = re.sub(r'\[\[.*?\]\]', '', text)
    
    # 5. 处理外部链接
    text = re.sub(r'\[https?://[^\s\]]+\s+([^\]]+)\]', r'\1', text)  # [URL 文本] → 文本
    text = re.sub(r'\[https?://[^\]]+\]', '', text)  # [URL] → 删除
    text = re.sub(r'https?://[^\s\n]+', '', text)  # 裸URL
    
    # ========== 第三阶段：HTML清理 ==========
    
    # 6. 移除丢弃的元素
    for tag in discardElements:
        pattern = re.compile(r'<\s*%s\b[^>]*>.*?<\s*/\s*%s>' % (tag, tag), re.DOTALL | re.IGNORECASE)
        text = pattern.sub('', text)
    
    # 7. 移除自闭合标签
    for tag in selfClosingTags:
        pattern = re.compile(r'<\s*%s\b[^/]*/\s*>' % tag, re.DOTALL | re.IGNORECASE)
        text = pattern.sub('', text)
    
    # 8. 移除忽略的标签（保留内容）
    for tag in ignoredTags:
        text = re.sub(r'<\s*%s\b[^>]*>' % tag, '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/\s*%s>' % tag, '', text, flags=re.IGNORECASE)
    
    # 9. 移除剩余的HTML标签
    text = re.sub(r'<[^>]*>', '', text)
    
    # ========== 第四阶段：格式清理 ==========
    
    # 10. 处理粗体/斜体
    text = re.sub(r"'''''([^']*?)'''''", r'\1', text)  # 粗斜体
    text = re.sub(r"'''(.*?)'''", r'\1', text)  # 粗体
    text = re.sub(r"''([^']*)''", r'\1', text)  # 斜体
    
    # 11. HTML实体解码
    text = unescape(text)
    text = unescape(text)  # 二次解码处理 &amp;nbsp; 等
    
    # ========== 第五阶段：中文特化清理 ==========
    
    # 12. 移除多语言标记
    text = re.sub(r'-zh-[^:]*:[^-]*-', '', text)
    text = re.sub(r'-\{[^}]*\}-', '', text)
    
    # 13. 移除引用标记和学术标识
    text = re.sub(r'\[\d+\]', '', text)  # [1], [2] 等
    text = re.sub(r'ISBN\s*[\d\-X]+', '', text)
    text = re.sub(r'DOI\s*:?\s*[\d\./]+', '', text)
    text = re.sub(r'\d{4}\s*reprint\.?', '', text)
    
    # 14. 移除图片描述词汇
    image_words = r'\b(thumb|thumbnail|right|left|center|frame|frameless|border|upright|\d+px|缩略图|右|左|居中|框架)\b'
    text = re.sub(image_words, '', text, flags=re.IGNORECASE)
    
    # 15. 移除行内公式和赋值
    text = re.sub(r'\([^\)]*=[^\)]*\)', '', text)  # (x = y)
    text = re.sub(r'\b[\w\-\+\*/^]{1,20}\s*=\s*[\w\-\+\*/^]{1,20}\b', '', text)  # a=b
    
    # ========== 第六阶段：智能过滤 ==========
    
    # 16. 按行过滤
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 跳过章节标题格式
        if re.match(r'=+.*?=+', line):
            continue
            
        # 跳过明显的图片描述
        if (len(line) < 200 and 
            line.count('，') > 2 and
            not any(char in line for char in ['。', '！', '？'])):
            continue
        
        # 跳过过短的行
        if len(line) < 15:
            continue
        
        # 跳过非中文为主的行
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', line))
        if chinese_chars < len(line) * 0.3:
            continue
        
        # 跳过列表项
        if re.match(r'^\s*[*#:;•·]', line):
            continue
            
        cleaned_lines.append(line)
    
    # 17. 重新组合文本
    text = ' '.join(cleaned_lines)
    
    # ========== 第七阶段：最终清理 ==========
    
    # 18. 标点符号规范化
    text = re.sub(r'\s*([，。！？；：])\s*', r'\1', text)
    text = re.sub(r'([。！？])\s*([^\s])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    
    # 19. 移除残留符号
    text = re.sub(r'[|{}()（）\[\]<>]', '', text)
    text = re.sub(r'[，。]{2,}', '。', text)
    text = re.sub(r'^\s*[。，！？；：]\s*', '', text)  # 开头的标点
    
    # 20. 最终质量检查
    text = text.strip()
    
    # 如果文本太短或中文比例太低，返回空
    if (len(text) < 100 or 
        len(re.findall(r'[\u4e00-\u9fff]', text)) < len(text) * 0.5 or
        len(re.findall(r'[\u4e00-\u9fff]', text)) < 50):
        return ""
    
    return text

def process_dump(dump_path: str, output_path: str, sample_path: str, sample_size=1000, max_articles=None):
    """处理维基百科dump文件"""
    try:
        from mwxml import Dump
    except ImportError:
        print("错误: 需要安装 mwxml 库")
        print("请运行: pip install mwxml")
        sys.exit(1)
    
    import time
    start_time = time.time()
    
    # 统计变量
    sample_count = 0
    processed_pages = 0
    valid_articles = 0
    total_text_length = 0
    total_chinese_chars = 0
    text_lengths = []  # 用于计算平均长度
    chinese_ratios = []  # 用于计算中文比例
    
    print(f"开始处理: {dump_path}")
    print(f"输出文件: {output_path}")
    print(f"样例文件: {sample_path} (前{sample_size}条)")
    if max_articles:
        print(f"最大处理文章数: {max_articles}")
    print("-" * 50)
    
    with bz2.open(dump_path, 'rb') as f, \
         open(output_path, 'w', encoding='utf-8') as out_f, \
         open(sample_path, 'w', encoding='utf-8') as sample_f:
        
        dump = Dump.from_file(f)
        
        for page in dump:
            processed_pages += 1
            
            # 跳过非主命名空间或重定向页
            if page.namespace != 0 or page.redirect:
                continue
            
            # 处理页面的最新版本
            for revision in page:
                text = revision.text or ""
                cleaned = clean_text(text)
                
                # 跳过清洗后为空或过短的文本
                if not cleaned or len(cleaned) < 100:
                    continue
                
                # 统计文本质量指标
                text_length = len(cleaned)
                chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', cleaned))
                chinese_ratio = chinese_chars / text_length if text_length > 0 else 0
                
                total_text_length += text_length
                total_chinese_chars += chinese_chars
                text_lengths.append(text_length)
                chinese_ratios.append(chinese_ratio)
                
                # 构建JSON对象
                obj = {
                    "text": cleaned,
                    "meta": {
                        "title": page.title,
                        "id": page.id,
                        "length": text_length,
                        "chinese_ratio": round(chinese_ratio, 3)
                    }
                }
                json_line = json.dumps(obj, ensure_ascii=False)
                
                # 写入主文件
                out_f.write(json_line + '\n')
                valid_articles += 1
                
                # 写入样例文件
                if sample_count < sample_size:
                    sample_f.write(json_line + '\n')
                    sample_count += 1
                
                # 检查是否达到最大文章数限制
                if max_articles and valid_articles >= max_articles:
                    print(f"\n已达到最大文章数限制: {max_articles}")
                    break
                
                break  # 只处理最新版本
            
            # 如果达到最大文章数限制，退出外层循环
            if max_articles and valid_articles >= max_articles:
                break
            
            # 定期输出进度
            if processed_pages % 1000 == 0:
                elapsed = time.time() - start_time
                pages_per_min = processed_pages / (elapsed / 60) if elapsed > 0 else 0
                print(f"已处理页面: {processed_pages:,}, 有效文章: {valid_articles:,}, "
                      f"样例: {sample_count}, 速度: {pages_per_min:.0f}页/分钟")
                out_f.flush()
                sample_f.flush()
    
    # 计算最终统计数据
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("📊 处理完成 - 效果评估报告")
    print("=" * 60)
    
    # 基本统计
    print(f"📈 基本统计:")
    print(f"  总页面数: {processed_pages:,}")
    print(f"  有效文章: {valid_articles:,}")
    print(f"  样例数量: {sample_count}")
    if max_articles:
        print(f"  文章数限制: {max_articles:,} {'(已达到)' if valid_articles >= max_articles else '(未达到)'}")
    print(f"  过滤比例: {((processed_pages - valid_articles) / processed_pages * 100):.1f}%")
    print(f"  处理时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"  处理速度: {processed_pages / (total_time / 60):.0f}页/分钟")
    
    if valid_articles > 0:
        # 文本质量指标
        avg_length = sum(text_lengths) / len(text_lengths)
        avg_chinese_ratio = sum(chinese_ratios) / len(chinese_ratios)
        
        # 长度分布统计
        short_texts = sum(1 for l in text_lengths if l < 500)
        medium_texts = sum(1 for l in text_lengths if 500 <= l < 2000)
        long_texts = sum(1 for l in text_lengths if l >= 2000)
        
        # 中文比例分布
        high_chinese = sum(1 for r in chinese_ratios if r >= 0.8)
        medium_chinese = sum(1 for r in chinese_ratios if 0.5 <= r < 0.8)
        low_chinese = sum(1 for r in chinese_ratios if r < 0.5)
        
        print(f"\n📋 质量指标:")
        print(f"  平均文本长度: {avg_length:.0f}字符")
        print(f"  平均中文比例: {avg_chinese_ratio:.1%}")
        print(f"  总字符数: {total_text_length:,}")
        print(f"  总中文字符: {total_chinese_chars:,}")
        
        print(f"\n📊 长度分布:")
        print(f"  短文本 (<500字符): {short_texts:,} ({short_texts/valid_articles:.1%})")
        print(f"  中等文本 (500-2000字符): {medium_texts:,} ({medium_texts/valid_articles:.1%})")
        print(f"  长文本 (>2000字符): {long_texts:,} ({long_texts/valid_articles:.1%})")
        
        print(f"\n🈳 中文比例分布:")
        print(f"  高中文比例 (≥80%): {high_chinese:,} ({high_chinese/valid_articles:.1%})")
        print(f"  中等中文比例 (50-80%): {medium_chinese:,} ({medium_chinese/valid_articles:.1%})")
        print(f"  低中文比例 (<50%): {low_chinese:,} ({low_chinese/valid_articles:.1%})")
        

    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="Wikipedia中文数据清洗工具 - 专为大模型预训练优化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理完整dump文件
  python clean.py zhwiki-20250601-pages-articles-multistream1.xml-p1p187712.bz2
  
  # 快速测试：只处理1000篇文章
  python clean.py dump.xml.bz2 --max-articles 1000
  
  # 自定义所有参数
  python clean.py dump.xml.bz2 --output clean.jsonl --sample sample.jsonl --sample-size 500 --max-articles 2000
        """
    )
    
    parser.add_argument(
        "dump_path",
        type=Path,
        help="维基百科dump文件路径 (*.xml.bz2)"
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        type=Path,
        default="zhwiki_cleaned.jsonl",
        help="输出JSONL文件路径 (默认: zhwiki_cleaned.jsonl)"
    )
    parser.add_argument(
        "--sample",
        dest="sample_path", 
        type=Path,
        default="sample_1000.jsonl",
        help="样例文件路径 (默认: sample_1000.jsonl)"
    )
    parser.add_argument(
        "--sample-size",
        dest="sample_size",
        type=int,
        default=1000,
        help="样例数量 (默认: 1000)"
    )
    parser.add_argument(
        "--max-articles",
        dest="max_articles",
        type=int,
        help="最大处理文章数"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Wikipedia中文清洗工具 v2.0"
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not args.dump_path.exists():
        print(f"错误: 文件不存在: {args.dump_path}")
        sys.exit(1)
    
    # 创建输出目录
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.sample_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        process_dump(
            str(args.dump_path),
            str(args.output_path), 
            str(args.sample_path),
            args.sample_size,
            args.max_articles
        )
    except KeyboardInterrupt:
        print("\n用户中断处理")
        sys.exit(1)
    except Exception as e:
        print(f"处理出错: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()