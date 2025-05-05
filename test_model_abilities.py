import requests
import json
import time
import random
import pandas as pd
import os

# VLLM服务的URL和模型名称
VLLM_URL = "http://14.103.133.112:8001/v1/chat/completions"
MODEL_NAME = "xDAN-L2-32b-Reasoning-Agent-Test"

# 测试用例示例
def test_tool_calling_ability(test_case):
    """测试模型的工具调用能力"""
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": test_case["messages"],
        "temperature": 0.0,  # 使用低温度以获得确定性输出
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(VLLM_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            model_output = result["choices"][0]["message"]["content"]
            
            # 验证输出
            return validate_output(model_output, test_case["expected_format"])
        else:
            print(f"请求失败: {response.status_code}")
            print(response.text)
            return False, f"HTTP错误: {response.status_code}"
    except Exception as e:
        print(f"请求异常: {str(e)}")
        return False, f"请求异常: {str(e)}"

def validate_output(output, expected_format):
    """验证模型输出是否符合预期格式"""
    
    if expected_format == "json":
        try:
            # 尝试解析为JSON
            json_output = json.loads(output)
            
            # 检查是否包含next字段
            if "next" in json_output:
                valid_values = ["researcher", "reporter", "browser", "searx_agent", "mira_agent", "finance_agent", "FINISH"]
                if json_output["next"] in valid_values:
                    return True, json_output
                else:
                    return False, f"'next'字段值无效: {json_output['next']}"
            else:
                return False, "JSON输出缺少'next'字段"
        except json.JSONDecodeError:
            return False, "输出不是有效的JSON格式"
    
    elif expected_format == "tool_call":
        # 检查是否包含<think>和<tool_call>标签
        has_think = "<think>" in output and "</think>" in output
        has_tool_call = "<tool_call>" in output and "</tool_call>" in output
        
        if has_think and has_tool_call:
            # 提取工具调用部分
            tool_call_start = output.find("<tool_call>") + len("<tool_call>")
            tool_call_end = output.find("</tool_call>")
            tool_call_content = output[tool_call_start:tool_call_end].strip()
            
            try:
                # 尝试解析工具调用内容为JSON
                # 可能有多个工具调用，每行一个JSON
                tool_calls = []
                for line in tool_call_content.strip().split("\n"):
                    if line.strip():
                        tool_call_json = json.loads(line.strip())
                        tool_calls.append(tool_call_json)
                
                if tool_calls:
                    return True, tool_calls
                else:
                    return False, "工具调用内容为空"
            except json.JSONDecodeError as e:
                return False, f"工具调用内容不是有效的JSON格式: {str(e)}"
        else:
            missing_tags = []
            if not has_think:
                missing_tags.append("<think>")
            if not has_tool_call:
                missing_tags.append("<tool_call>")
            return False, f"缺少标签: {', '.join(missing_tags)}"
    
    elif expected_format == "response":
        # 检查是否包含<think>和<response>标签
        has_think = "<think>" in output and "</think>" in output
        has_response = "<response>" in output and "</response>" in output
        
        if has_think and has_response:
            # 提取响应部分
            response_start = output.find("<response>") + len("<response>")
            response_end = output.find("</response>")
            response_content = output[response_start:response_end].strip()
            
            if response_content:
                return True, response_content
            else:
                return False, "响应内容为空"
        else:
            missing_tags = []
            if not has_think:
                missing_tags.append("<think>")
            if not has_response:
                missing_tags.append("<response>")
            return False, f"缺少标签: {', '.join(missing_tags)}"
    
    return False, "不支持的验证格式"

# 运行多个测试用例
def run_tests(test_cases):
    """运行所有测试用例"""
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"运行测试用例 {i+1}/{len(test_cases)}: {test_case['name']}")
        
        start_time = time.time()
        success, output = test_tool_calling_ability(test_case)
        end_time = time.time()
        
        result = {
            "name": test_case["name"],
            "success": success,
            "output": output,
            "time_taken": end_time - start_time
        }
        
        results.append(result)
        
        print(f"测试结果: {'成功' if success else '失败'}")
        if not success:
            print(f"失败原因: {output}")
        print(f"耗时: {end_time - start_time:.2f}秒")
        print("-" * 50)
    
    return results

# 从Parquet文件加载测试样本
def load_test_samples_from_parquet(file_path, num_samples=10):
    """从Parquet文件中加载测试样本"""
    
    try:
        # 读取Parquet文件
        df = pd.read_parquet(file_path)
        
        # 随机选择样本
        if len(df) > num_samples:
            sample_indices = random.sample(range(len(df)), num_samples)
            samples = df.iloc[sample_indices]
        else:
            samples = df
        
        # 转换为测试用例格式
        test_cases = []
        for i, (_, sample) in enumerate(samples.iterrows()):
            # 提取提示和预期输出
            if isinstance(sample.get("prompt"), list):
                prompt = sample.get("prompt", [])
            else:
                # 如果prompt不是列表，尝试解析
                try:
                    prompt = json.loads(sample.get("prompt", "[]"))
                except:
                    prompt = [{"role": "user", "content": str(sample.get("prompt", ""))}]
            
            # 获取奖励模型信息
            if isinstance(sample.get("reward_model"), dict):
                reward_model = sample.get("reward_model", {})
            else:
                try:
                    reward_model = json.loads(sample.get("reward_model", "{}"))
                except:
                    reward_model = {}
            
            ground_truth = reward_model.get("ground_truth", "")
            
            # 确定预期输出格式
            if "<tool_call>" in str(ground_truth):
                expected_format = "tool_call"
            elif "<response>" in str(ground_truth):
                expected_format = "response"
            else:
                expected_format = "unknown"
            
            test_case = {
                "name": f"数据集样本 #{i+1}",
                "messages": prompt,
                "expected_format": expected_format,
                "ground_truth": ground_truth
            }
            
            test_cases.append(test_case)
        
        return test_cases
    except Exception as e:
        print(f"加载Parquet文件失败: {str(e)}")
        return []

# 计算评估指标
def calculate_metrics(results):
    """计算评估指标"""
    
    if not results:
        return {"错误": "没有测试结果"}
    
    total = len(results)
    success_count = sum(1 for result in results if result["success"])
    
    # 按类型分类
    tool_call_tests = [r for r in results if r.get("expected_format") == "tool_call"]
    json_tests = [r for r in results if r.get("expected_format") == "json"]
    response_tests = [r for r in results if r.get("expected_format") == "response"]
    
    # 计算各类型成功率
    tool_call_success = sum(1 for r in tool_call_tests if r["success"]) / len(tool_call_tests) if tool_call_tests else 0
    json_success = sum(1 for r in json_tests if r["success"]) / len(json_tests) if json_tests else 0
    response_success = sum(1 for r in response_tests if r["success"]) / len(response_tests) if response_tests else 0
    
    # 计算平均响应时间
    avg_time = sum(r["time_taken"] for r in results) / total
    
    metrics = {
        "总体准确率": success_count / total * 100,
        "工具调用准确率": tool_call_success * 100 if tool_call_tests else "N/A",
        "JSON输出准确率": json_success * 100 if json_tests else "N/A",
        "直接响应准确率": response_success * 100 if response_tests else "N/A",
        "平均响应时间": avg_time,
        "测试用例总数": total,
        "成功用例数": success_count
    }
    
    return metrics

# 自定义测试用例
def create_custom_test_cases():
    """创建自定义测试用例"""
    
    test_cases = [
        {
            "name": "简单工具调用测试 - 是否为回文",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个有用的AI助手。你可以使用以下工具：\n\n1. 名称: is_palindrome\n描述: 检查一个字符串是否为回文（正着读和倒着读一样）\n参数: {\"text\": {\"description\": \"要检查的文本\", \"type\": \"string\"}}\n\n请按照以下格式输出：\n<think>你的思考过程</think>\n<tool_call>\n{\"name\": \"工具名称\", \"parameters\": {参数}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "\"level\"这个词是回文吗？"
                }
            ],
            "expected_format": "tool_call"
        },
        {
            "name": "数学计算工具测试",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个有用的AI助手。你可以使用以下工具：\n\n1. 名称: solve_quadratic\n描述: 求解二次方程ax²+bx+c=0\n参数: {\"a\": {\"description\": \"二次项系数\", \"type\": \"number\"}, \"b\": {\"description\": \"一次项系数\", \"type\": \"number\"}, \"c\": {\"description\": \"常数项\", \"type\": \"number\"}}\n\n请按照以下格式输出：\n<think>你的思考过程</think>\n<tool_call>\n{\"name\": \"工具名称\", \"parameters\": {参数}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "求解方程 x²+5x+6=0"
                }
            ],
            "expected_format": "tool_call"
        },
        {
            "name": "代理选择测试",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个监督者，协调专业工作者团队完成任务。你的团队包括：[reporter, researcher, browser, searx_agent, mira_agent, finance_agent]。\n\n对于每个用户请求，你将：\n1. 分析请求并确定哪个工作者最适合处理\n2. 仅以JSON对象格式响应：{\"next\": \"worker_name\"}\n3. 审查他们的响应并：\n   - 如果需要更多工作，选择下一个工作者（例如，{\"next\": \"researcher\"}）\n   - 当任务完成时，回复{\"next\": \"FINISH\"}\n\n始终以仅包含'next'键和单个值的有效JSON对象响应：工作者名称或'FINISH'。"
                },
                {
                    "role": "user",
                    "content": "帮我搜下最近的国际要闻"
                }
            ],
            "expected_format": "json"
        },
        {
            "name": "搜索工具测试",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个有用的AI助手。你可以使用以下工具：\n\n1. 名称: search\n描述: 在互联网上搜索信息\n参数: {\"query\": {\"description\": \"搜索查询\", \"type\": \"string\"}}\n\n请按照以下格式输出：\n<think>你的思考过程</think>\n<tool_call>\n{\"name\": \"工具名称\", \"parameters\": {参数}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "最近有什么重要的国际新闻？"
                }
            ],
            "expected_format": "tool_call"
        },
        {
            "name": "多工具调用测试",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个有用的AI助手。你可以使用以下工具：\n\n1. 名称: is_anagram\n描述: 检查两个词是否互为字谜（包含相同的字母但顺序不同）\n参数: {\"word1\": {\"description\": \"第一个词\", \"type\": \"string\"}, \"word2\": {\"description\": \"第二个词\", \"type\": \"string\"}}\n\n请按照以下格式输出：\n<think>你的思考过程</think>\n<tool_call>\n{\"name\": \"工具名称\", \"parameters\": {参数}}\n</tool_call>"
                },
                {
                    "role": "user",
                    "content": "\"Dormitory\"和\"Dirty room\"是字谜吗？\"Conversation\"和\"Voices rant on\"呢？"
                }
            ],
            "expected_format": "tool_call"
        }
    ]
    
    return test_cases

# 保存测试结果
def save_results(results, metrics, filename="测试结果.json"):
    """保存测试结果到文件"""
    
    output = {
        "metrics": metrics,
        "results": results
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"测试结果已保存到: {filename}")

# 主函数
def main():
    """主函数"""
    
    print(f"开始测试模型: {MODEL_NAME}")
    print(f"VLLM服务URL: {VLLM_URL}")
    print("-" * 50)
    
    # 创建自定义测试用例
    custom_test_cases = create_custom_test_cases()
    print(f"已创建 {len(custom_test_cases)} 个自定义测试用例")
    
    # 运行自定义测试
    print("\n运行自定义测试用例...")
    custom_results = run_tests(custom_test_cases)
    
    # 从数据集加载测试样本
    dataset_path = "/Users/gumpcehng/CascadeProjects/xDAN-Agentic-RL-ToolRL/dataset/rlla_4k/test.parquet"
    if os.path.exists(dataset_path):
        print(f"\n从数据集加载测试样本: {dataset_path}")
        dataset_test_cases = load_test_samples_from_parquet(dataset_path, num_samples=10)
        print(f"已加载 {len(dataset_test_cases)} 个数据集测试样本")
        
        if dataset_test_cases:
            # 运行数据集测试
            print("\n运行数据集测试用例...")
            dataset_results = run_tests(dataset_test_cases)
            all_results = custom_results + dataset_results
        else:
            print("未能加载数据集测试样本，跳过数据集测试")
            all_results = custom_results
    else:
        print(f"数据集文件不存在: {dataset_path}")
        all_results = custom_results
    
    # 计算并打印指标
    print("\n计算评估指标...")
    metrics = calculate_metrics(all_results)
    
    print("\n测试结果汇总:")
    for name, value in metrics.items():
        if isinstance(value, float):
            if "时间" in name:
                print(f"{name}: {value:.2f}秒")
            else:
                print(f"{name}: {value:.2f}%")
        else:
            print(f"{name}: {value}")
    
    # 保存测试结果
    save_results(all_results, metrics)

if __name__ == "__main__":
    main()
