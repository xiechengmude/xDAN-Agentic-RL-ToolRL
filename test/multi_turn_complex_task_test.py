import sys
import os
import json
from typing import List, Dict, Any, Optional

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入 API 推理模块
from inference.api_inference import run_chat, parse_tool_calls, parse_thinking, parse_response

# 模拟工具执行函数
def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> str:
    """
    模拟执行工具并返回结果
    
    参数:
        tool_name: 工具名称
        parameters: 工具参数
        
    返回:
        工具执行结果
    """
    print(f"\n执行工具: {tool_name}")
    print(f"参数: {json.dumps(parameters, ensure_ascii=False, indent=2)}")
    
    if tool_name == "search":
        query = parameters.get("query", "")
        if "天气" in query or "weather" in query:
            if "北京" in query or "Beijing" in query:
                return "北京今天天气晴朗，气温22-28°C，空气质量良好，适合户外活动。"
            elif "上海" in query or "Shanghai" in query:
                return "上海今天多云，气温20-26°C，有轻微污染，建议减少户外活动时间。"
            elif "广州" in query or "Guangzhou" in query:
                return "广州今天阵雨，气温24-30°C，湿度较高，外出请携带雨具。"
            else:
                return "查询的城市天气信息：晴天，气温20-25°C。"
        elif "餐厅" in query or "restaurant" in query:
            return "附近有以下餐厅：1. 老北京烤鸭（评分4.8星），2. 江南小馆（评分4.5星），3. 粤海渔村（评分4.7星）"
        elif "电影" in query or "movie" in query:
            return "最近热映电影：1.《流浪地球3》，2.《你好，李焕英2》，3.《速度与激情11》"
        else:
            return f"搜索结果：关于'{query}'的信息。这是一个模拟的搜索结果。"
    
    elif tool_name == "get_location":
        return "当前位置：北京市海淀区中关村"
    
    elif tool_name == "get_time":
        return "当前时间：2025年5月11日 16:51"
    
    elif tool_name == "book_restaurant":
        restaurant = parameters.get("restaurant", "")
        time = parameters.get("time", "")
        people = parameters.get("people", 1)
        return f"已为您预订{restaurant}，时间：{time}，{people}人用餐。预订成功，订单号：R2025051101"
    
    elif tool_name == "book_movie":
        movie = parameters.get("movie", "")
        cinema = parameters.get("cinema", "")
        time = parameters.get("time", "")
        seats = parameters.get("seats", 1)
        return f"已为您预订电影《{movie}》，影院：{cinema}，时间：{time}，{seats}个座位。预订成功，订单号：M2025051102"
    
    else:
        return f"工具 {tool_name} 执行结果：这是一个模拟的工具执行结果。"

# 定义测试用的工具列表
def get_test_tools() -> List[Dict[str, Any]]:
    """获取测试用的工具列表"""
    return [
        {
            "name": "search", 
            "description": "搜索信息，可以查询天气、餐厅、电影等信息", 
            "parameters": {"properties": {"query": {"type": "string"}}}
        },
        {
            "name": "get_location", 
            "description": "获取当前位置信息", 
            "parameters": {"properties": {}}
        },
        {
            "name": "get_time", 
            "description": "获取当前时间", 
            "parameters": {"properties": {}}
        },
        {
            "name": "book_restaurant", 
            "description": "预订餐厅", 
            "parameters": {"properties": {
                "restaurant": {"type": "string"},
                "time": {"type": "string"},
                "people": {"type": "integer"}
            }}
        },
        {
            "name": "book_movie", 
            "description": "预订电影票", 
            "parameters": {"properties": {
                "movie": {"type": "string"},
                "cinema": {"type": "string"},
                "time": {"type": "string"},
                "seats": {"type": "integer"}
            }}
        }
    ]

# 运行多轮对话测试
def run_multi_turn_test(initial_query: str, max_turns: int = 5):
    """
    运行多轮对话测试
    
    参数:
        initial_query: 初始查询
        max_turns: 最大对话轮数
    """
    tools = get_test_tools()
    messages = [{"role": "user", "content": initial_query}]
    
    print(f"\n===== 开始多轮对话测试 =====")
    print(f"初始查询: {initial_query}")
    print(f"可用工具: {', '.join([t['name'] for t in tools])}")
    
    for turn in range(1, max_turns + 1):
        print(f"\n----- 对话轮次 {turn} -----")
        
        # 调用模型进行推理
        output, tool_calls = run_chat(tools, messages)
        
        # 解析模型输出
        thinking = parse_thinking(output)
        response = parse_response(output)
        
        print(f"\n模型思考:")
        print(thinking if thinking else "无思考内容")
        
        if tool_calls:
            print(f"\n模型调用工具:")
            for call in tool_calls:
                tool_name = call.get("name", "")
                parameters = call.get("parameters", {})
                
                # 执行工具并获取结果
                tool_result = execute_tool(tool_name, parameters)
                
                # 将工具执行结果添加到对话历史
                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": tool_result
                })
                
                print(f"工具结果: {tool_result}")
        
        if response:
            print(f"\n模型响应:")
            print(response)
            
            # 将模型响应添加到对话历史
            messages.append({
                "role": "assistant",
                "content": output
            })
            
            # 如果模型给出了最终响应，自动生成下一轮对话内容
            if turn < max_turns:
                # 预定义的后续问题列表
                follow_up_questions = [
                    "预订成功了吗？我可以知道预订详情吗？",
                    "餐厅有什么特色菜？可以推荐几道菜吗？",
                    "餐厅附近有什么好玩的地方吗？我们吃完饭想去玩一下"
                ]
                
                # 根据当前轮次选择问题
                if turn <= len(follow_up_questions):
                    user_input = follow_up_questions[turn-1]
                else:
                    # 如果问题用完了，就结束对话
                    print("\n预定义的问题已用完，结束对话")
                    break
                
                print(f"\n下一轮用户输入: {user_input}")
                messages.append({"role": "user", "content": user_input})
            
        else:
            # 如果没有响应但有工具调用，继续下一轮
            print("\n模型未给出响应，继续执行工具调用...")
    
    print(f"\n===== 对话测试结束 =====")

# 主函数
if __name__ == "__main__":
    # 测试复杂任务：查询天气和餐厅，并预订餐厅
    complex_query = "我想了解今天北京的天气，然后找一家好评的餐厅并预订今晚7点的位子，两个人用餐"
    
    # 运行测试
    run_multi_turn_test(complex_query)
