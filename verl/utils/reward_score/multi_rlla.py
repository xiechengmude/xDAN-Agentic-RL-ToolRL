# Copyright 2024 xDAN Agentic RL ToolRL Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
import os
from collections import Counter, defaultdict
import numpy as np
from typing import List, Dict, Any, Tuple

# 从现有模块导入通用功能
from .rlla import match_score, customize_format_reward_func, customize_length_reward_func


def extract_thinking(message: Dict[str, Any]) -> str:
    """从助手消息中提取思考内容"""
    if message.get("role") != "assistant":
        return ""
    
    content = message.get("content", "")
    if "<think>" in content and "</think>" in content:
        return content.split("<think>")[1].split("</think>")[0].strip()
    return ""


def extract_tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从助手消息中提取工具调用"""
    if message.get("role") != "assistant":
        return []
    
    content = message.get("content", "")
    if "<tool_call>" not in content or "</tool_call>" not in content:
        return []
    
    try:
        tool_call_text = content.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        tool_calls = [json.loads(tool) for tool in tool_call_text.split("\n") if tool.strip()]
        return tool_calls
    except Exception:
        return []


def extract_response(message: Dict[str, Any]) -> str:
    """从助手消息中提取最终回复"""
    if message.get("role") != "assistant":
        return ""
    
    content = message.get("content", "")
    if "<response>" in content and "</response>" in content:
        return content.split("<response>")[1].split("</response>")[0].strip()
    return ""


def evaluate_tool_selection(conversation: List[Dict[str, Any]], 
                           gt_conversation: List[Dict[str, Any]]) -> float:
    """评估工具选择的正确性
    
    评估模型在每轮交互中是否选择了正确的工具，并且输入了正确的参数
    """
    score = 0.0
    total_turns = 0
    
    # 提取模型选择的工具调用
    model_turns = [msg for msg in conversation if msg.get("role") == "assistant"]
    gt_turns = [msg for msg in gt_conversation if msg.get("role") == "assistant"]
    
    # 确保我们只比较相同数量的轮次
    turn_count = min(len(model_turns), len(gt_turns))
    
    for i in range(turn_count):
        model_tools = extract_tool_calls(model_turns[i])
        gt_tools = extract_tool_calls(gt_turns[i])
        
        if not model_tools and not gt_tools:
            continue
            
        total_turns += 1
        
        # 如果这轮没有工具调用但期望有，或者有工具调用但期望没有，得分为0
        if bool(model_tools) != bool(gt_tools):
            continue
            
        # 计算工具名称匹配得分
        model_tool_names = [tool.get("name", "") for tool in model_tools]
        gt_tool_names = [tool.get("name", "") for tool in gt_tools]
        name_score = match_score(model_tool_names, gt_tool_names)
        
        # 计算工具参数匹配得分
        param_score = 0.0
        for gt_tool in gt_tools:
            gt_name = gt_tool.get("name", "")
            gt_params = gt_tool.get("parameters", {})
            
            # 查找匹配的模型工具
            matching_tools = [t for t in model_tools if t.get("name", "") == gt_name]
            if matching_tools:
                model_tool = matching_tools[0]  # 取第一个匹配的工具
                model_params = model_tool.get("parameters", {})
                
                # 参数名称匹配
                param_key_score = match_score(list(gt_params.keys()), list(model_params.keys()))
                
                # 参数值匹配
                param_value_score = sum(1.0 for k, v in gt_params.items() 
                                     if k in model_params and model_params[k] == v) / max(len(gt_params), 1)
                
                param_score += (param_key_score + param_value_score) / 2.0
        
        # 标准化参数得分
        if gt_tools:
            param_score /= len(gt_tools)
        
        # 结合名称和参数得分
        turn_score = (name_score + param_score) / 2.0
        score += turn_score
    
    # 标准化最终得分
    return score / max(total_turns, 1)


def evaluate_info_utilization(conversation: List[Dict[str, Any]]) -> float:
    """评估模型如何利用前轮工具返回的信息
    
    检查模型是否在思考过程中参考了之前工具调用的结果
    """
    score = 0.0
    tool_calls = []
    tool_results = []
    
    # 收集所有工具调用及其结果
    for i, msg in enumerate(conversation):
        if msg.get("role") == "assistant":
            tool_calls.extend(extract_tool_calls(msg))
        elif msg.get("role") == "tool" and i > 0:
            tool_results.append({
                "name": msg.get("name", ""),
                "content": msg.get("content", "")
            })
    
    if not tool_results:
        return 1.0  # 没有工具结果需要利用
    
    # 收集所有助手在工具调用后的思考内容
    post_tool_thoughts = []
    for i, msg in enumerate(conversation):
        if msg.get("role") == "assistant" and i > 0:
            # 检查是否在某个工具返回后
            if i > 0 and conversation[i-1].get("role") == "tool":
                thinking = extract_thinking(msg)
                if thinking:
                    post_tool_thoughts.append(thinking)
    
    # 如果没有后续思考，无法评估
    if not post_tool_thoughts:
        return 0.5
    
    # 评估每个工具结果在后续思考中的利用情况
    for tool_result in tool_results:
        tool_name = tool_result["name"]
        content = tool_result["content"]
        
        # 尝试解析JSON内容
        try:
            if isinstance(content, str) and (content.startswith("{") or content.startswith("[")):
                result_data = json.loads(content)
                # 提取关键信息（如数字、状态、关键词等）
                key_info = extract_key_info(result_data)
            else:
                key_info = [content[:50]]  # 使用前50个字符作为关键信息
        except:
            key_info = [content[:50]]
        
        # 检查关键信息是否出现在后续思考中
        info_matches = 0
        for thought in post_tool_thoughts:
            for info in key_info:
                if str(info) in thought:
                    info_matches += 1
                    break
        
        # 计算这个工具结果的利用率
        tool_score = min(info_matches / max(len(post_tool_thoughts), 1), 1.0)
        score += tool_score
    
    # 标准化最终得分
    return score / len(tool_results)


def extract_key_info(data: Any) -> List[str]:
    """从工具结果数据中提取关键信息"""
    key_info = []
    
    if isinstance(data, dict):
        # 优先提取状态、结果、错误等信息
        for key in ["status", "result", "error", "data", "value", "score"]:
            if key in data:
                key_info.append(str(data[key]))
        
        # 提取所有数值型数据
        for k, v in data.items():
            if isinstance(v, (int, float)):
                key_info.append(f"{k}:{v}")
            elif isinstance(v, (dict, list)):
                key_info.extend(extract_key_info(v))
    
    elif isinstance(data, list) and len(data) > 0:
        # 对于列表，采样前3个元素（如果存在）
        for item in data[:min(3, len(data))]:
            key_info.extend(extract_key_info(item))
    
    else:
        # 基本类型则直接添加字符串表示
        key_info.append(str(data)[:50])
    
    return key_info[:10]  # 限制返回的关键信息数量


def evaluate_reasoning_chain(conversation: List[Dict[str, Any]]) -> float:
    """评估整个工具调用链的推理质量
    
    检查工具调用之间的逻辑关系，以及思考过程的合理性
    """
    score = 0.0
    chain_coherence = 0.0
    
    # 提取所有助手消息的思考内容
    thoughts = []
    for msg in conversation:
        if msg.get("role") == "assistant":
            thinking = extract_thinking(msg)
            if thinking:
                thoughts.append(thinking)
    
    if len(thoughts) <= 1:
        return 0.75  # 只有一轮思考，给予中等偏上分数
    
    # 评估思考过程的连贯性
    for i in range(1, len(thoughts)):
        prev_thought = thoughts[i-1]
        curr_thought = thoughts[i]
        
        # 检查当前思考是否参考了前一轮思考的结论
        coherence = check_thought_coherence(prev_thought, curr_thought)
        chain_coherence += coherence
    
    # 标准化连贯性得分
    chain_coherence_score = chain_coherence / (len(thoughts) - 1)
    
    # 评估工具调用的顺序逻辑
    tools_order_score = evaluate_tools_order(conversation)
    
    # 综合得分 (权重可调整)
    score = 0.6 * chain_coherence_score + 0.4 * tools_order_score
    return score


def check_thought_coherence(prev_thought: str, curr_thought: str) -> float:
    """检查两个思考过程之间的连贯性"""
    # 这里使用一个简单的启发式方法：检查关键句子的重叠度
    
    # 将文本分割为句子
    prev_sentences = [s.strip() for s in re.split(r'[.!?。！？]', prev_thought) if s.strip()]
    curr_sentences = [s.strip() for s in re.split(r'[.!?。！？]', curr_thought) if s.strip()]
    
    # 提取前一个思考的后半部分（结论部分）
    prev_conclusion = " ".join(prev_sentences[-min(3, len(prev_sentences)):])
    
    # 提取当前思考的前半部分（引用前文部分）
    curr_intro = " ".join(curr_sentences[:min(3, len(curr_sentences))])
    
    # 计算语义重叠 (简化为词汇重叠)
    prev_words = set(prev_conclusion.lower().split())
    curr_words = set(curr_intro.lower().split())
    
    if not prev_words or not curr_words:
        return 0.5
    
    overlap = len(prev_words.intersection(curr_words))
    coherence = overlap / max(len(prev_words.union(curr_words)), 1)
    
    return min(coherence * 2, 1.0)  # 放大重叠得分，但最高为1.0


def evaluate_tools_order(conversation: List[Dict[str, Any]]) -> float:
    """评估工具调用顺序的合理性"""
    tool_calls = []
    
    # 收集所有工具调用
    for msg in conversation:
        if msg.get("role") == "assistant":
            tools = extract_tool_calls(msg)
            if tools:
                tool_calls.append({
                    "tools": tools,
                    "thinking": extract_thinking(msg)
                })
    
    if len(tool_calls) <= 1:
        return 0.8  # 只有一次工具调用，给予较高分数
    
    # 评估工具调用之间的逻辑关系
    order_score = 0.0
    
    for i in range(1, len(tool_calls)):
        prev_tools = tool_calls[i-1]["tools"]
        prev_thinking = tool_calls[i-1]["thinking"]
        curr_tools = tool_calls[i]["tools"]
        curr_thinking = tool_calls[i]["thinking"]
        
        # 检查当前工具调用是否建立在前一个工具结果的基础上
        dependency_score = check_tool_dependency(
            prev_tools, prev_thinking, curr_tools, curr_thinking)
        order_score += dependency_score
    
    # 标准化得分
    return order_score / (len(tool_calls) - 1)


def check_tool_dependency(prev_tools, prev_thinking, curr_tools, curr_thinking) -> float:
    """检查工具调用之间的依赖关系"""
    # 简单启发式方法：检查当前思考过程是否提到了之前的工具名称
    
    prev_tool_names = [tool.get("name", "") for tool in prev_tools]
    curr_tool_names = [tool.get("name", "") for tool in curr_tools]
    
    # 检查是否有提及前一个工具名称
    tool_mention_score = 0.0
    for name in prev_tool_names:
        if name in curr_thinking:
            tool_mention_score += 1.0
    
    tool_mention_score = min(tool_mention_score / max(len(prev_tool_names), 1), 1.0)
    
    # 检查工具类型序列是否符合常见模式 (如搜索->分析->执行)
    tool_pattern_score = check_tool_pattern(prev_tool_names, curr_tool_names)
    
    # 综合得分
    return 0.7 * tool_mention_score + 0.3 * tool_pattern_score


def check_tool_pattern(prev_tools, curr_tools) -> float:
    """检查工具调用序列是否符合常见模式"""
    # 这里可以实现更复杂的逻辑，当前使用简化版本
    
    # 工具类型分类
    tool_categories = {
        # 信息获取类工具
        "get": ["get_", "fetch_", "retrieve_", "search_", "find_", "lookup_"],
        # 分析类工具
        "analyze": ["analyze_", "calculator", "compute_", "evaluate_", "assess_"],
        # 执行类工具
        "execute": ["create_", "update_", "delete_", "set_", "apply_", "execute_"]
    }
    
    # 识别工具类型
    def get_tool_category(tool_name):
        tool_name = tool_name.lower()
        for category, prefixes in tool_categories.items():
            if any(tool_name.startswith(prefix) for prefix in prefixes):
                return category
        return "other"
    
    # 检查是否符合 get->analyze->execute 模式
    pattern_score = 0.5  # 默认中等分数
    
    prev_categories = [get_tool_category(tool) for tool in prev_tools]
    curr_categories = [get_tool_category(tool) for tool in curr_tools]
    
    # 检查一些常见的良好模式
    if "get" in prev_categories and "analyze" in curr_categories:
        pattern_score = 0.8
    elif "analyze" in prev_categories and "execute" in curr_categories:
        pattern_score = 0.8
    elif "get" in prev_categories and "execute" in curr_categories:
        pattern_score = 0.6  # 跳过分析步骤，得分较低
    
    return pattern_score


def evaluate_problem_solving(conversation: List[Dict[str, Any]]) -> float:
    """评估最终是否解决了用户问题
    
    检查最后的回复是否有针对性地回答了用户的原始问题
    """
    # 找到用户的第一个问题
    user_query = ""
    for msg in conversation:
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break
    
    if not user_query:
        return 0.5  # 没有找到用户问题
    
    # 找到最后一个助手回复
    final_response = ""
    for msg in reversed(conversation):
        if msg.get("role") == "assistant":
            final_response = extract_response(msg)
            if final_response:
                break
    
    if not final_response:
        return 0.0  # 没有找到最终回复
    
    # 简单启发式评估回复的相关性和完整性
    # 1. 长度检查 - 回复不应过短
    length_score = min(len(final_response) / 100, 1.0)
    
    # 2. 关键词匹配 - 回复应包含用户问题中的关键词
    user_keywords = extract_keywords(user_query)
    response_keywords = extract_keywords(final_response)
    
    keyword_overlap = len(user_keywords.intersection(response_keywords))
    keyword_score = keyword_overlap / max(len(user_keywords), 1)
    
    # 3. 结构评分 - 回复应有清晰的结构（如分点说明）
    structure_score = check_response_structure(final_response)
    
    # 4. 回复完整性 - 检查回复是否包含完整的内容
    completeness_score = check_response_completeness(final_response)
    
    # 综合得分 (权重可调整)
    score = (0.1 * length_score + 
             0.3 * keyword_score + 
             0.3 * structure_score + 
             0.3 * completeness_score)
    
    return score


def extract_keywords(text: str) -> set:
    """从文本中提取关键词"""
    # 移除标点符号
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    
    # 移除停用词 (简化版)
    stop_words = {"a", "an", "the", "is", "are", "was", "were", "be", "been", 
                  "being", "in", "on", "at", "to", "for", "with", "by", "about", 
                  "against", "between", "into", "through", "during", "before", 
                  "after", "above", "below", "from", "up", "down", "of", "off", 
                  "over", "under", "again", "further", "then", "once", "here", 
                  "there", "when", "where", "why", "how", "all", "any", "both", 
                  "each", "few", "more", "most", "other", "some", "such", "no", 
                  "nor", "not", "only", "own", "same", "so", "than", "too", "very", 
                  "s", "t", "can", "will", "just", "don", "should", "now"}
    
    words = [word for word in text.split() if word not in stop_words and len(word) > 2]
    return set(words)


def check_response_structure(response: str) -> float:
    """检查回复的结构清晰度"""
    # 检查是否有列表标记
    has_bullets = bool(re.search(r'(\n\s*[-•*]\s+|\n\s*\d+\.\s+)', response))
    
    # 检查是否有段落结构
    paragraphs = [p for p in response.split("\n\n") if p.strip()]
    has_paragraphs = len(paragraphs) >= 2
    
    # 检查是否有小标题
    has_headers = bool(re.search(r'\n\s*[A-Z][^.!?]*[:：]', response))
    
    # 结构得分
    if has_bullets and (has_paragraphs or has_headers):
        return 1.0
    elif has_bullets or has_headers:
        return 0.8
    elif has_paragraphs:
        return 0.6
    else:
        return 0.4


def check_response_completeness(response: str) -> float:
    """检查回复的完整性"""
    # 检查是否包含总结性语句
    has_conclusion = bool(re.search(r'(总之|总结|综上|In summary|In conclusion|Overall)', response))
    
    # 检查是否包含数字或百分比 (表明有具体数据)
    has_numbers = bool(re.search(r'\d+(\.\d+)?%?', response))
    
    # 检查是否包含推荐或建议
    has_recommendations = bool(re.search(r'(建议|推荐|recommend|suggest|consider|option|策略|方案|approach)', 
                                         response.lower()))
    
    # 完整性得分
    completeness = 0.5  # 基础分
    if has_conclusion:
        completeness += 0.2
    if has_numbers:
        completeness += 0.15
    if has_recommendations:
        completeness += 0.15
    
    return min(completeness, 1.0)


def normalize_reward(reward: float, max_reward: float, min_reward: float) -> float:
    """将奖励值归一化到指定范围"""
    # 首先归一化到[0,1]
    normalized = max(0.0, min(1.0, reward))
    
    # 然后缩放到[min_reward, max_reward]
    scaled = min_reward + normalized * (max_reward - min_reward)
    
    return scaled


def customize_multi_turn_reward(completions, answers, step, 
                               max_possible_reward=3.0, min_possible_reward=-3.0, **kwargs):
    """计算多轮对话奖励的主函数 (与框架接口兼容)"""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for response, answer in zip(responses, answers):
        # 从字符串解析对话历史
        try:
            # 这里需要根据实际数据格式进行解析
            # 假设每个回复和答案都是JSON字符串，包含完整对话历史
            conversation = parse_conversation_from_response(response)
            gt_conversation = parse_conversation_from_answer(answer)
            
            # 计算多轮奖励
            reward = compute_multi_turn_reward(
                conversation, 
                gt_conversation,
                max_reward=max_possible_reward,
                min_reward=min_possible_reward
            )
            
            rewards.append(reward)
        except Exception as e:
            print(f"Error calculating multi-turn reward: {str(e)}")
            rewards.append(min_possible_reward)
    
    print("\n======= Reward for multi-turn dialogue =======")
    print("Reward function for multi-turn dialogue is called ...")
    print(rewards)
    return rewards


def parse_conversation_from_response(response_str: str) -> List[Dict[str, Any]]:
    """从响应字符串解析对话历史 (需根据实际数据格式自定义)"""
    # 这个函数需要根据实际数据格式定制
    # 示例实现假设响应是一个包含对话历史的JSON字符串
    try:
        if isinstance(response_str, str) and response_str.strip().startswith("["):
            return json.loads(response_str)
        else:
            # 如果不是JSON格式，尝试从特定标记解析
            conversation = []
            
            # 解析用户消息
            user_parts = re.findall(r'<user>(.*?)</user>', response_str, re.DOTALL)
            for part in user_parts:
                conversation.append({"role": "user", "content": part.strip()})
            
            # 解析助手消息 (包含思考、工具调用和回复)
            assistant_parts = re.findall(r'<think>(.*?)</think>(?:\n<tool_call>(.*?)</tool_call>)?(?:\n<response>(.*?)</response>)?', 
                                         response_str, re.DOTALL)
            for thinking, tool_call, response in assistant_parts:
                content = f"<think>{thinking.strip()}</think>"
                if tool_call:
                    content += f"\n<tool_call>{tool_call.strip()}</tool_call>"
                if response:
                    content += f"\n<response>{response.strip()}</response>"
                
                conversation.append({"role": "assistant", "content": content})
            
            # 解析工具反馈
            tool_parts = re.findall(r'<obs>(.*?)</obs>', response_str, re.DOTALL)
            for part in tool_parts:
                # 尝试解析工具名称和内容
                tool_match = re.search(r'You have made the tool call (.*?)\. Execution returns: (.*)', part.strip())
                if tool_match:
                    tool_name = tool_match.group(1)
                    tool_content = tool_match.group(2)
                    conversation.append({"role": "tool", "name": tool_name, "content": tool_content})
            
            return conversation
    except Exception as e:
        print(f"Error parsing conversation from response: {str(e)}")
        return []


def parse_conversation_from_answer(answer_str: str) -> List[Dict[str, Any]]:
    """从标准答案字符串解析对话历史 (需根据实际数据格式自定义)"""
    # 与parse_conversation_from_response类似，但用于解析标准答案
    # 实际实现需要根据数据格式定制
    return parse_conversation_from_response(answer_str)


def compute_multi_turn_reward(
    conversation_history: List[Dict[str, Any]],
    gt_conversation: List[Dict[str, Any]],
    max_reward: float = 3.0,
    min_reward: float = -3.0
) -> float:
    """计算多轮工具调用的奖励"""
    # 环境变量控制：不同组件奖励权重
    tool_selection_weight = float(os.getenv("MULTI_TOOL_SELECTION_WEIGHT", "0.35"))
    info_utilization_weight = float(os.getenv("MULTI_INFO_UTILIZATION_WEIGHT", "0.25"))
    reasoning_chain_weight = float(os.getenv("MULTI_REASONING_CHAIN_WEIGHT", "0.25"))
    problem_solving_weight = float(os.getenv("MULTI_PROBLEM_SOLVING_WEIGHT", "0.15"))
    
    # 1. 工具选择评分 - 每轮工具选择是否正确
    tool_selection_reward = evaluate_tool_selection(conversation_history, gt_conversation)
    
    # 2. 信息利用评分 - 评估模型如何利用前轮工具返回信息
    info_utilization_reward = evaluate_info_utilization(conversation_history)
    
    # 3. 推理链评分 - 评估整个工具调用链是否合理
    reasoning_chain_reward = evaluate_reasoning_chain(conversation_history)
    
    # 4. 问题解决评分 - 评估是否最终解决了用户问题
    problem_solving_reward = evaluate_problem_solving(conversation_history)
    
    # 打印详细得分，便于调试
    print(f"Tool Selection Score: {tool_selection_reward:.4f}")
    print(f"Info Utilization Score: {info_utilization_reward:.4f}")
    print(f"Reasoning Chain Score: {reasoning_chain_reward:.4f}")
    print(f"Problem Solving Score: {problem_solving_reward:.4f}")
    
    # 综合计算最终奖励 (加权平均)
    normalized_reward = (
        tool_selection_weight * tool_selection_reward +
        info_utilization_weight * info_utilization_reward +
        reasoning_chain_weight * reasoning_chain_reward + 
        problem_solving_weight * problem_solving_reward
    )
    
    # 归一化到指定范围
    final_reward = min_reward + normalized_reward * (max_reward - min_reward)
    
    print(f"Final Multi-turn Reward: {final_reward:.4f}")
    return final_reward