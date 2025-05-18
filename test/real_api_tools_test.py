import sys
import os
import json
import requests
import datetime
import time
import random
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入 API 推理模块
from inference.api_inference import run_chat, parse_tool_calls, parse_thinking, parse_response

# 加载环境变量
load_dotenv()

# 真实工具执行函数
def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> str:
    """
    执行真实工具并返回结果
    
    参数:
        tool_name: 工具名称
        parameters: 工具参数
        
    返回:
        工具执行结果
    """
    print(f"\n执行工具: {tool_name}")
    print(f"参数: {json.dumps(parameters, ensure_ascii=False, indent=2)}")
    
    try:
        if tool_name == "get_weather":
            return get_weather(parameters.get("city", "北京"))
        
        elif tool_name == "search_web":
            return search_web(parameters.get("query", ""))
        
        elif tool_name == "get_time":
            return get_current_time(parameters.get("timezone", "Asia/Shanghai"))
        
        elif tool_name == "calculate":
            return calculate_expression(parameters.get("expression", ""))
        
        elif tool_name == "translate":
            return translate_text(
                parameters.get("text", ""), 
                parameters.get("source_lang", "auto"), 
                parameters.get("target_lang", "zh")
            )
        
        elif tool_name == "get_news":
            return get_news(parameters.get("category", "general"))
        
        elif tool_name == "get_stock_price":
            return get_stock_price(parameters.get("symbol", ""))
        
        elif tool_name == "generate_image_description":
            return generate_image_description(parameters.get("image_url", ""))
        
        else:
            return f"错误：未知工具 {tool_name}"
    
    except Exception as e:
        return f"工具执行错误: {str(e)}"

# 1. 天气查询工具
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    try:
        api_key = os.getenv("OPENWEATHER_API_KEY", "")
        if not api_key:
            # 如果没有 API 密钥，返回模拟数据
            weather_conditions = ["晴朗", "多云", "阵雨", "小雨", "大雨", "雷阵雨", "小雪", "大雪"]
            temp_min = random.randint(15, 25)
            temp_max = temp_min + random.randint(3, 8)
            condition = random.choice(weather_conditions)
            return f"{city}今天天气{condition}，气温{temp_min}-{temp_max}°C。(模拟数据，请配置OPENWEATHER_API_KEY环境变量获取真实数据)"
        
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=zh_cn"
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200:
            weather = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            feels_like = data["main"]["feels_like"]
            humidity = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]
            
            return f"{city}当前天气：{weather}，温度{temp}°C，体感温度{feels_like}°C，湿度{humidity}%，风速{wind_speed}m/s"
        else:
            return f"获取天气信息失败：{data.get('message', '未知错误')}"
    
    except Exception as e:
        return f"天气查询出错：{str(e)}"

# 2. 网络搜索工具
def search_web(query: str) -> str:
    """使用 SerpAPI 进行网络搜索"""
    try:
        api_key = os.getenv("SERPAPI_API_KEY", "")
        if not api_key:
            return f"搜索查询：{query}\n(请配置SERPAPI_API_KEY环境变量获取真实搜索结果)"
        
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": api_key,
            "engine": "google",
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if "error" in data:
            return f"搜索出错：{data['error']}"
        
        results = []
        if "organic_results" in data:
            for i, result in enumerate(data["organic_results"][:3], 1):
                title = result.get("title", "无标题")
                snippet = result.get("snippet", "无描述")
                link = result.get("link", "无链接")
                results.append(f"{i}. {title}\n   {snippet}\n   链接：{link}")
        
        if results:
            return f"关于'{query}'的搜索结果：\n\n" + "\n\n".join(results)
        else:
            return f"没有找到关于"{query}"的搜索结果"
    
    except Exception as e:
        return f"搜索出错：{str(e)}"

# 3. 获取当前时间工具
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取指定时区的当前时间"""
    try:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"当前时间（{timezone}）：{current_time}"
    except Exception as e:
        return f"获取时间出错：{str(e)}"

# 4. 计算器工具
def calculate_expression(expression: str) -> str:
    """计算数学表达式"""
    try:
        # 注意：eval 函数在生产环境中可能存在安全风险
        # 这里仅作为示例，实际应用中应使用更安全的方法
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算出错：{str(e)}"

# 5. 文本翻译工具
def translate_text(text: str, source_lang: str = "auto", target_lang: str = "zh") -> str:
    """翻译文本"""
    try:
        api_key = os.getenv("DEEPL_API_KEY", "")
        if not api_key:
            # 简单的模拟翻译
            if target_lang == "zh" and not all('\u4e00' <= char <= '\u9fff' for char in text if char.isalpha()):
                return f"翻译结果：「{text}」→ 这是一个模拟的中文翻译结果。(请配置DEEPL_API_KEY环境变量获取真实翻译)"
            elif target_lang == "en":
                return f"翻译结果：「{text}」→ This is a simulated English translation. (请配置DEEPL_API_KEY环境变量获取真实翻译)"
            else:
                return f"翻译结果：「{text}」→ 模拟翻译到{target_lang}语言。(请配置DEEPL_API_KEY环境变量获取真实翻译)"
        
        url = "https://api-free.deepl.com/v2/translate"
        headers = {
            "Authorization": f"DeepL-Auth-Key {api_key}"
        }
        data = {
            "text": [text],
            "target_lang": target_lang.upper()
        }
        
        if source_lang != "auto":
            data["source_lang"] = source_lang.upper()
        
        response = requests.post(url, headers=headers, data=data)
        result = response.json()
        
        if "translations" in result and result["translations"]:
            translated_text = result["translations"][0]["text"]
            detected_source_lang = result["translations"][0].get("detected_source_language", source_lang)
            return f"翻译结果（{detected_source_lang} → {target_lang}）：{translated_text}"
        else:
            return f"翻译失败：{json.dumps(result)}"
    
    except Exception as e:
        return f"翻译出错：{str(e)}"

# 6. 新闻获取工具
def get_news(category: str = "general") -> str:
    """获取指定类别的新闻"""
    try:
        api_key = os.getenv("NEWSAPI_API_KEY", "")
        if not api_key:
            categories = {
                "general": ["国内政治新闻", "国际关系动态", "社会热点事件"],
                "business": ["股市行情分析", "企业并购消息", "经济政策解读"],
                "technology": ["AI技术突破", "新款智能手机发布", "网络安全事件"],
                "entertainment": ["电影票房排行", "明星活动动态", "音乐新专辑发布"],
                "sports": ["足球比赛结果", "NBA季后赛进展", "奥运会筹备情况"],
                "science": ["太空探索新发现", "医学研究突破", "环境科学进展"],
                "health": ["疫情防控措施", "健康生活方式", "医疗技术创新"]
            }
            
            news_titles = categories.get(category, categories["general"])
            news_list = [f"{i+1}. {title}" for i, title in enumerate(news_titles)]
            
            return f"{category}类别新闻：\n" + "\n".join(news_list) + "\n(模拟数据，请配置NEWSAPI_API_KEY环境变量获取真实新闻)"
        
        url = f"https://newsapi.org/v2/top-headlines"
        params = {
            "apiKey": api_key,
            "category": category,
            "language": "zh",
            "pageSize": 5
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get("status") == "ok" and data.get("articles"):
            news_list = []
            for i, article in enumerate(data["articles"][:5], 1):
                title = article.get("title", "无标题")
                description = article.get("description", "无描述")
                source = article.get("source", {}).get("name", "未知来源")
                news_list.append(f"{i}. {title}\n   来源：{source}\n   简介：{description}")
            
            return f"{category}类别新闻：\n\n" + "\n\n".join(news_list)
        else:
            return f"获取新闻失败：{data.get('message', '未知错误')}"
    
    except Exception as e:
        return f"获取新闻出错：{str(e)}"

# 7. 股票价格查询工具
def get_stock_price(symbol: str) -> str:
    """获取股票价格信息"""
    try:
        api_key = os.getenv("ALPHAVANTAGE_API_KEY", "")
        if not api_key:
            # 模拟股票数据
            price = round(random.uniform(50, 500), 2)
            change = round(random.uniform(-20, 20), 2)
            change_percent = round(change / (price - change) * 100, 2)
            change_sign = "+" if change >= 0 else ""
            
            return f"股票 {symbol} 当前价格：${price}\n涨跌：{change_sign}{change} ({change_sign}{change_percent}%)\n(模拟数据，请配置ALPHAVANTAGE_API_KEY环境变量获取真实股票数据)"
        
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if "Global Quote" in data and data["Global Quote"]:
            quote = data["Global Quote"]
            price = quote.get("05. price", "未知")
            change = quote.get("09. change", "未知")
            change_percent = quote.get("10. change percent", "未知")
            
            return f"股票 {symbol} 当前价格：${price}\n涨跌：{change} ({change_percent})"
        else:
            return f"获取股票信息失败：{json.dumps(data)}"
    
    except Exception as e:
        return f"获取股票信息出错：{str(e)}"

# 8. 图像描述生成工具
def generate_image_description(image_url: str) -> str:
    """生成图像描述"""
    try:
        api_key = os.getenv("AZURE_VISION_API_KEY", "")
        endpoint = os.getenv("AZURE_VISION_ENDPOINT", "")
        
        if not api_key or not endpoint:
            return f"图像URL：{image_url}\n描述：这是一张图片。(模拟描述，请配置AZURE_VISION_API_KEY和AZURE_VISION_ENDPOINT环境变量获取真实图像描述)"
        
        vision_url = f"{endpoint}/vision/v3.1/analyze"
        
        headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "Content-Type": "application/json"
        }
        
        params = {
            "visualFeatures": "Description,Tags",
            "language": "zh"
        }
        
        body = {
            "url": image_url
        }
        
        response = requests.post(vision_url, headers=headers, params=params, json=body)
        result = response.json()
        
        if "description" in result and "captions" in result["description"]:
            description = result["description"]["captions"][0]["text"]
            confidence = result["description"]["captions"][0]["confidence"]
            
            tags = []
            if "tags" in result:
                tags = [tag["name"] for tag in result["tags"][:5]]
            
            return f"图像描述（置信度：{confidence:.2f}）：{description}\n标签：{', '.join(tags)}"
        else:
            return f"生成图像描述失败：{json.dumps(result)}"
    
    except Exception as e:
        return f"生成图像描述出错：{str(e)}"

# 定义测试用的工具列表
def get_real_tools() -> List[Dict[str, Any]]:
    """获取真实工具列表"""
    return [
        {
            "name": "get_weather", 
            "description": "获取指定城市的天气信息", 
            "parameters": {"properties": {"city": {"type": "string"}}}
        },
        {
            "name": "search_web", 
            "description": "使用搜索引擎查询信息", 
            "parameters": {"properties": {"query": {"type": "string"}}}
        },
        {
            "name": "get_time", 
            "description": "获取当前时间", 
            "parameters": {"properties": {"timezone": {"type": "string"}}}
        },
        {
            "name": "calculate", 
            "description": "计算数学表达式", 
            "parameters": {"properties": {"expression": {"type": "string"}}}
        },
        {
            "name": "translate", 
            "description": "翻译文本", 
            "parameters": {"properties": {
                "text": {"type": "string"},
                "source_lang": {"type": "string"},
                "target_lang": {"type": "string"}
            }}
        },
        {
            "name": "get_news", 
            "description": "获取指定类别的新闻", 
            "parameters": {"properties": {"category": {"type": "string"}}}
        },
        {
            "name": "get_stock_price", 
            "description": "获取股票价格信息", 
            "parameters": {"properties": {"symbol": {"type": "string"}}}
        },
        {
            "name": "generate_image_description", 
            "description": "生成图像描述", 
            "parameters": {"properties": {"image_url": {"type": "string"}}}
        }
    ]

# 运行多轮对话测试
def run_real_api_test(initial_query: str, max_turns: int = 5):
    """
    运行使用真实API的多轮对话测试
    
    参数:
        initial_query: 初始查询
        max_turns: 最大对话轮数
    """
    tools = get_real_tools()
    messages = [{"role": "user", "content": initial_query}]
    
    print(f"\n===== 开始真实API多轮对话测试 =====")
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
                    "能给我详细解释一下这些信息吗？",
                    "你能用这些工具帮我做更复杂的任务吗？比如查询股票信息并分析趋势",
                    "请帮我翻译一段文字：Hello, how are you today? I would like to learn Chinese.",
                    "今天北京和上海的天气怎么样？哪个城市更适合户外活动？"
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
    # 测试复杂任务
    complex_query = "帮我查询一下今天北京的天气，然后计算一下23*45的结果，最后帮我翻译'人工智能'到英文"
    
    # 运行测试
    run_real_api_test(complex_query)
