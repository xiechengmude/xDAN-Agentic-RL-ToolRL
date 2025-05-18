import json
import requests
import time
import argparse
import sys
import random
from typing import List, Dict, Any, Tuple, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 配置
API_URL = "http://60.50.228.238:62735/v1"
MODEL = "Qwen2.5-7B-Instruct_ZeroSearch"
TEMPERATURE = 0.7
MAX_RETRIES = 3
RETRY_BACKOFF = 2
TEST_MODE = False  # 设置为True时使用模拟响应

# 定义系统提示 - 使用模型的原始系统提示
SYSTEM_PROMPT = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <search> query </search>, and it "
    "will return the top searched results between <information> and </information>. "
    "You can search as many times as you want. If you find no further external knowledge "
    "needed, you can directly provide the answer inside <answer> and </answer> without "
    "detailed illustrations. For example, <answer> Beijing </answer>."
)

# 创建一个带有重试机制的会话
def create_retry_session(retries=MAX_RETRIES, backoff_factor=RETRY_BACKOFF):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# 生成模拟响应（用于测试模式）
def generate_mock_response(query: str) -> str:
    # 模拟思考过程
    thinking = f"""我需要分析用户的问题："{query}"。
这个问题需要我搜索相关信息。让我先思考一下我已知的信息，然后决定是否需要搜索更多内容。"""
    
    # 模拟搜索查询
    search_query = query.replace("？", "").replace("?", "").strip()
    
    # 模拟答案
    answer = f"""根据搜索结果，我可以回答这个问题了。{query.replace("？", "").replace("?", "")}的答案是相关信息。"""
    
    # 随机决定是否需要搜索
    if random.random() > 0.3:
        return f"<think>{thinking}</think>\n<search>{search_query}</search>\n<answer>{answer}</answer>"
    else:
        return f"<think>{thinking}</think>\n<answer>{answer}</answer>"

# API调用函数
def call_api(prompt: str, max_tokens: int = 16384, temperature: float = TEMPERATURE) -> str:
    """调用远程API获取模型响应"""
    # 测试模式下返回模拟响应
    if TEST_MODE:
        print("测试模式：生成模拟响应")
        time.sleep(1)  # 模拟网络延迟
        return generate_mock_response(prompt)
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"{SYSTEM_PROMPT} Question: {prompt}"}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    session = create_retry_session()
    
    try:
        print(f"正在调用API: {API_URL}/chat/completions")
        print(f"请求模型: {MODEL}, 温度: {temperature}")
        response = session.post(f"{API_URL}/chat/completions", headers=headers, json=payload, timeout=30)
        print(f"API响应状态码: {response.status_code}")
        response.raise_for_status()
        result = response.json()
        
        # 只打印响应的关键部分
        if 'choices' in result and len(result['choices']) > 0:
            print("成功获取API响应")
            
        # 从正确的位置获取响应内容
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except requests.exceptions.RequestException as e:
        print(f"API请求错误: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_content = e.response.json()
                print(f"错误详情: {json.dumps(error_content, ensure_ascii=False, indent=2)}")
            except:
                print(f"响应内容: {e.response.text}")
        
        # 在API错误时切换到测试模式
        print("切换到测试模式以继续执行")
        return generate_mock_response(prompt)
    except Exception as e:
        print(f"其他错误: {e}")
        
        # 在其他错误时也切换到测试模式
        print("切换到测试模式以继续执行")
        return generate_mock_response(prompt)

# 从模型输出中提取思考内容
def parse_thinking(output: str) -> Optional[str]:
    if "<think>" in output and "</think>" in output:
        return output.split("<think>")[1].split("</think>")[0].strip()
    return None

# 从模型输出中提取搜索查询
def parse_search_queries(output: str) -> List[str]:
    queries = []
    start_pos = 0
    while True:
        start_tag = output.find("<search>", start_pos)
        if start_tag == -1:
            break
        end_tag = output.find("</search>", start_tag)
        if end_tag == -1:
            break
        query = output[start_tag + len("<search>"):end_tag].strip()
        queries.append(query)
        start_pos = end_tag + len("</search>")
    return queries

# 从模型输出中提取最终答案
def parse_answer(output: str) -> Optional[str]:
    if "<answer>" in output and "</answer>" in output:
        return output.split("<answer>")[1].split("</answer>")[0].strip()
    return None

# 模拟搜索工具的执行
def execute_search(query: str) -> str:
    """
    模拟执行搜索，返回搜索结果
    在实际应用中，这里应该调用真实的搜索API
    """
    print(f"执行搜索: {query}")
    # 这里可以替换为实际的搜索API调用
    # 为了演示，我们返回一些模拟的搜索结果
    time.sleep(1)  # 模拟网络延迟
    
    # 根据查询内容生成不同的模拟结果
    topics = {
        "量子计算": [
            "IBM在2023年发布了127量子比特的Eagle处理器，量子体积达到128",
            "谷歌声称在2023年实现了量子霸权的新里程碑，使用76量子比特的Sycamore处理器解决了特定问题",
            "量子纠错技术取得重大突破，使量子计算更加稳定可靠",
            "量子算法在药物发现领域应用取得进展，加速了新药研发过程"
        ],
        "人工智能": [
            "大型语言模型GPT-4在多模态理解和生成方面取得突破",
            "自监督学习方法在减少训练数据需求方面取得进展",
            "AI辅助科学发现领域出现新应用，如在材料科学和生物学中",
            "边缘AI技术使设备能够在不依赖云计算的情况下进行复杂推理"
        ],
        "气候变化": [
            "最新IPCC报告指出全球温度上升速度快于预期",
            "碳捕获技术在工业规模应用取得进展",
            "可再生能源成本持续下降，太阳能发电成本创历史新低",
            "极端天气事件频率和强度增加，与气候变化有明显关联"
        ]
    }
    
    # 查找最匹配的主题
    matched_topic = None
    for topic in topics:
        if topic in query:
            matched_topic = topic
            break
    
    if matched_topic:
        results = topics[matched_topic]
    else:
        # 默认结果
        results = [
            f"关于'{query}'的最新研究表明这是一个快速发展的领域",
            f"专家们对'{query}'持有不同观点，有些认为...",
            f"最近的'{query}'相关会议讨论了几个关键问题",
            f"'{query}'领域的技术创新正在加速"
        ]
    
    # 随机选择2-4个结果
    num_results = random.randint(2, min(4, len(results)))
    selected_results = random.sample(results, num_results)
    
    return "- " + "\n- ".join(selected_results)

# 主要推理函数
def run_search_chat(query: str, max_search_rounds: int = 3) -> Tuple[str, List[str], str]:
    """
    使用远程API进行搜索增强对话推理
    
    参数:
        query: 用户查询
        max_search_rounds: 最大搜索轮数
        
    返回:
        完整的模型输出、搜索查询列表和最终答案
    """
    current_round = 0
    final_output = ""
    all_search_queries = []
    final_answer = ""
    
    # 初始提示
    current_prompt = query
    
    while current_round < max_search_rounds:
        print(f"\n===== 第 {current_round + 1} 轮 =====")
        
        # 调用API获取响应
        generated_text = call_api(current_prompt)
        print(f"模型输出:\n{generated_text}")
        
        # 解析思考、搜索查询和答案
        thinking = parse_thinking(generated_text)
        if thinking:
            print(f"\n思考过程:\n{thinking}")
        
        search_queries = parse_search_queries(generated_text)
        if search_queries:
            print(f"\n搜索查询:\n{search_queries}")
            all_search_queries.extend(search_queries)
        
        answer = parse_answer(generated_text)
        if answer:
            print(f"\n最终答案:\n{answer}")
            final_answer = answer
        
        # 如果没有搜索查询或已有最终答案，结束循环
        if not search_queries or answer:
            final_output = generated_text
            break
        
        # 执行搜索并准备下一轮提示
        search_results = []
        for query in search_queries:
            result = execute_search(query)
            search_results.append(f"<information>\n{result}\n</information>")
        
        # 更新提示，包含原始查询和搜索结果
        current_prompt = f"{query}\n\n{''.join(search_results)}"
        
        current_round += 1
        print(f"完成第 {current_round} 轮搜索")
    
    return final_output, all_search_queries, final_answer

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Qwen2.5搜索增强测试')
    parser.add_argument('--query', type=str, default="请告诉我关于量子计算的最新进展",
                        help='用户查询')
    parser.add_argument('--rounds', type=int, default=3,
                        help='最大搜索轮数')
    parser.add_argument('--test', action='store_true',
                        help='使用测试模式（不调用实际API）')
    parser.add_argument('--model', type=str, default=MODEL,
                        help='使用的模型名称')
    parser.add_argument('--api_url', type=str, default=API_URL,
                        help='API URL')
    return parser.parse_args()

# 主函数
def main():
    args = parse_args()
    
    # 更新全局配置
    global TEST_MODE, MODEL, API_URL
    TEST_MODE = args.test
    MODEL = args.model
    API_URL = args.api_url
    
    print(f"\n===== Qwen2.5搜索增强测试 =====")
    print(f"模型: {MODEL}")
    print(f"API URL: {API_URL}")
    print(f"测试模式: {'启用' if TEST_MODE else '禁用'}")
    print(f"最大搜索轮数: {args.rounds}")
    print(f"用户查询: {args.query}")
    print("================================\n")
    
    # 运行搜索增强对话
    final_output, search_queries, final_answer = run_search_chat(args.query, args.rounds)
    
    # 输出结果摘要
    print("\n===== 结果摘要 =====")
    print(f"总搜索次数: {len(search_queries)}")
    if search_queries:
        print(f"搜索查询: {', '.join(search_queries)}")
    if final_answer:
        print(f"最终答案: {final_answer}")
    else:
        print("未获得最终答案")
    print("====================\n")

if __name__ == "__main__":
    main()
