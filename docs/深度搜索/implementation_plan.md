# TraceGenerator 增强实施计划

基于对 OpenAI DeepSearch 的分析和我们当前 TraceGenerator 的实现，本文档提出了近期优化的具体实施计划。

## 1. 搜索策略动态调整机制

当前的 `EnhancedTraceGenerator` 已经支持多轮交互和基本的后续搜索生成，但缺乏根据搜索结果质量动态调整策略的能力。

### 1.1 实施目标

- 实现搜索结果质量评估机制
- 根据评估结果动态调整搜索策略
- 支持从宽泛到精确的渐进式搜索

### 1.2 具体实现

```python
class SearchQualityEvaluator:
    """搜索结果质量评估器"""
    
    def evaluate(self, search_result, question, previous_results=None):
        """评估搜索结果质量
        
        返回:
            float: 质量得分 (0-1)
            dict: 评估详情
        """
        # 实现评估逻辑
        relevance_score = self._evaluate_relevance(search_result, question)
        coverage_score = self._evaluate_coverage(search_result, question)
        diversity_score = self._evaluate_diversity(search_result, previous_results)
        
        # 综合得分
        overall_score = 0.4 * relevance_score + 0.4 * coverage_score + 0.2 * diversity_score
        
        return overall_score, {
            "relevance": relevance_score,
            "coverage": coverage_score,
            "diversity": diversity_score,
            "overall": overall_score
        }
    
    def _evaluate_relevance(self, search_result, question):
        """评估相关性"""
        # 实现相关性评估逻辑
        pass
    
    def _evaluate_coverage(self, search_result, question):
        """评估覆盖度"""
        # 实现覆盖度评估逻辑
        pass
    
    def _evaluate_diversity(self, search_result, previous_results):
        """评估多样性"""
        # 实现多样性评估逻辑
        pass
```

在 `EnhancedTraceGenerator` 中集成质量评估器：

```python
def __init__(self, config: DataGenerationConfig):
    # 现有初始化代码...
    
    # 添加搜索质量评估器
    self.search_evaluator = SearchQualityEvaluator()
    
    # 添加搜索策略管理器
    self.search_strategy_manager = SearchStrategyManager()
```

实现搜索策略管理器：

```python
class SearchStrategyManager:
    """搜索策略管理器"""
    
    def __init__(self):
        self.strategies = {
            "broad": self._generate_broad_query,
            "focused": self._generate_focused_query,
            "specific": self._generate_specific_query,
            "alternative": self._generate_alternative_query
        }
        
        # 初始策略
        self.current_strategy = "broad"
        
    def adjust_strategy(self, quality_score, previous_strategies, question_complexity):
        """根据搜索质量调整策略"""
        if quality_score < 0.3:
            # 结果质量差，尝试完全不同的策略
            return "alternative"
        elif quality_score < 0.6:
            # 结果质量一般，尝试更聚焦的策略
            return "focused" if self.current_strategy == "broad" else "specific"
        else:
            # 结果质量好，继续深入
            return "specific"
    
    def generate_query(self, strategy, question, previous_queries, previous_results):
        """根据策略生成查询"""
        generator = self.strategies.get(strategy, self._generate_broad_query)
        return generator(question, previous_queries, previous_results)
    
    def _generate_broad_query(self, question, previous_queries, previous_results):
        """生成宽泛查询"""
        # 实现宽泛查询生成逻辑
        pass
    
    def _generate_focused_query(self, question, previous_queries, previous_results):
        """生成聚焦查询"""
        # 实现聚焦查询生成逻辑
        pass
    
    def _generate_specific_query(self, question, previous_queries, previous_results):
        """生成特定查询"""
        # 实现特定查询生成逻辑
        pass
    
    def _generate_alternative_query(self, question, previous_queries, previous_results):
        """生成替代查询"""
        # 实现替代查询生成逻辑
        pass
```

修改 `_generate_follow_up_search` 方法：

```python
def _generate_follow_up_search(self, question: str, steps: List[Dict[str, Any]]) -> str:
    """根据之前的步骤生成后续搜索查询"""
    # 提取之前的搜索查询和结果
    previous_queries = []
    previous_results = []
    
    for i, step in enumerate(steps):
        if step["type"] == "tool_call" and step["name"] in ["brightdata_search", "fetch_content"]:
            previous_queries.append(step["input"]["content"])
            
            # 找到对应的结果
            if i+1 < len(steps) and steps[i+1]["type"] == "tool_response":
                previous_results.append(steps[i+1]["result"])
    
    # 评估最近一次搜索结果的质量
    if previous_results:
        quality_score, details = self.search_evaluator.evaluate(
            previous_results[-1], 
            question,
            previous_results[:-1] if len(previous_results) > 1 else None
        )
        
        # 根据质量调整策略
        strategy = self.search_strategy_manager.adjust_strategy(
            quality_score,
            [q for q in previous_queries],
            self._estimate_question_complexity(question)
        )
        
        # 使用选定策略生成查询
        new_query = self.search_strategy_manager.generate_query(
            strategy,
            question,
            previous_queries,
            previous_results
        )
    else:
        # 首次搜索，使用宽泛策略
        new_query = self.search_strategy_manager.generate_query(
            "broad",
            question,
            [],
            []
        )
    
    return new_query
```

## 2. 错误处理和恢复机制

当前的实现在工具调用失败时缺乏健壮的错误处理和恢复机制。

### 2.1 实施目标

- 增强对各类错误的检测和处理
- 实现失败后的自动重试和策略调整
- 提供错误信息的结构化记录

### 2.2 具体实现

增强 `_execute_tool_call` 方法：

```python
def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """执行工具调用"""
    tool_name = tool_call["name"]
    tool_input = tool_call["input"]
    
    # 最大重试次数
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            if tool_name == self.search_tool.name:
                # 执行搜索
                result = self.search_tool.search(tool_input["content"])
                
                # 检查结果是否有效
                if self._is_valid_search_result(result):
                    return result
                else:
                    # 搜索结果无效，记录错误
                    error_msg = f"搜索结果无效: {str(result)}"
                    print(f"Warning: {error_msg}")
                    
                    # 如果是最后一次重试，返回错误信息
                    if retry_count == max_retries - 1:
                        return {
                            "error": error_msg,
                            "partial_result": result,
                            "retry_count": retry_count + 1
                        }
                    
                    # 否则调整输入并重试
                    tool_input["content"] = self._adjust_query_for_retry(
                        tool_input["content"], 
                        result, 
                        retry_count
                    )
            
            elif tool_name == "fetch_content" and self.content_fetcher:
                # 执行内容获取
                result = self.content_fetcher.fetch(tool_input["content"])
                
                # 检查结果是否有效
                if self._is_valid_content_result(result):
                    return result
                else:
                    # 内容获取结果无效，记录错误
                    error_msg = f"内容获取结果无效: {str(result)}"
                    print(f"Warning: {error_msg}")
                    
                    # 如果是最后一次重试，返回错误信息
                    if retry_count == max_retries - 1:
                        return {
                            "error": error_msg,
                            "partial_result": result,
                            "retry_count": retry_count + 1
                        }
                    
                    # 否则调整输入并重试
                    tool_input["content"] = self._adjust_url_for_retry(
                        tool_input["content"], 
                        result, 
                        retry_count
                    )
            else:
                return {"error": f"未知工具: {tool_name}"}
                
        except Exception as e:
            # 捕获异常，记录错误
            error_msg = f"工具调用异常: {str(e)}"
            print(f"Error: {error_msg}")
            import traceback
            traceback.print_exc()
            
            # 如果是最后一次重试，返回错误信息
            if retry_count == max_retries - 1:
                return {
                    "error": error_msg,
                    "exception": str(e),
                    "retry_count": retry_count + 1
                }
        
        # 增加重试计数
        retry_count += 1
        
        # 重试前等待
        time.sleep(1 * (retry_count + 1))  # 递增等待时间
    
    # 所有重试都失败，返回默认错误信息
    return {"error": "工具调用失败，已达到最大重试次数"}
```

添加辅助方法：

```python
def _is_valid_search_result(self, result):
    """检查搜索结果是否有效"""
    # 检查是否有错误
    if "error" in result:
        return False
    
    # 检查是否有搜索结果
    if "results" not in result or not result["results"]:
        return False
    
    # 检查结果是否有内容
    for item in result["results"]:
        if "title" in item and "snippet" in item:
            return True
    
    return False

def _is_valid_content_result(self, result):
    """检查内容获取结果是否有效"""
    # 检查是否有错误
    if "error" in result:
        return False
    
    # 检查是否有内容
    if "content" not in result or not result["content"]:
        return False
    
    return True

def _adjust_query_for_retry(self, query, failed_result, retry_count):
    """调整查询以重试"""
    if retry_count == 0:
        # 第一次重试，简化查询
        words = query.split()
        if len(words) > 3:
            return " ".join(words[:3])
        else:
            return query
    elif retry_count == 1:
        # 第二次重试，使用同义词或相关词
        return self._generate_alternative_query(query)
    else:
        # 最后重试，使用更通用的查询
        return self._generate_more_general_query(query)

def _adjust_url_for_retry(self, url, failed_result, retry_count):
    """调整URL以重试"""
    # 实现URL调整逻辑
    pass

def _generate_alternative_query(self, query):
    """生成替代查询"""
    # 实现替代查询生成逻辑
    pass

def _generate_more_general_query(self, query):
    """生成更通用的查询"""
    # 实现更通用查询生成逻辑
    pass
```

## 3. 多轮交互中的决策逻辑

当前的实现使用固定的交互轮次范围和随机决策，缺乏基于搜索结果和问题进展的智能决策。

### 3.1 实施目标

- 实现基于搜索进展的交互轮次决策
- 增强对问题解决程度的评估
- 支持智能的搜索终止条件

### 3.2 具体实现

添加问题解决进度评估器：

```python
class ProgressEvaluator:
    """问题解决进度评估器"""
    
    def evaluate(self, question, steps):
        """评估问题解决进度
        
        返回:
            float: 进度得分 (0-1)
            dict: 评估详情
        """
        # 实现评估逻辑
        information_coverage = self._evaluate_information_coverage(question, steps)
        answer_completeness = self._evaluate_answer_completeness(question, steps)
        
        # 综合得分
        overall_progress = 0.6 * information_coverage + 0.4 * answer_completeness
        
        return overall_progress, {
            "information_coverage": information_coverage,
            "answer_completeness": answer_completeness,
            "overall_progress": overall_progress
        }
    
    def _evaluate_information_coverage(self, question, steps):
        """评估信息覆盖度"""
        # 实现信息覆盖度评估逻辑
        pass
    
    def _evaluate_answer_completeness(self, question, steps):
        """评估答案完整度"""
        # 实现答案完整度评估逻辑
        pass
```

在 `EnhancedTraceGenerator` 中集成进度评估器：

```python
def __init__(self, config: DataGenerationConfig):
    # 现有初始化代码...
    
    # 添加进度评估器
    self.progress_evaluator = ProgressEvaluator()
```

修改 `generate_trace` 方法中的交互轮次决策逻辑：

```python
# 处理后续工具调用，直到达到目标轮次或问题解决
max_rounds = self.max_rounds
min_rounds = self.min_rounds
current_round = 1  # 已经完成了第一轮

while current_round < max_rounds:
    # 评估当前进度
    progress, details = self.progress_evaluator.evaluate(question, trace["steps"])
    
    # 如果已经达到最小轮次且进度足够高，可以提前结束
    if current_round >= min_rounds and progress > 0.85:
        print(f"问题已基本解决，提前结束交互。进度: {progress:.2f}")
        break
    
    # 如果进度不足且已接近最大轮次，可以适当延长
    if current_round >= max_rounds - 1 and progress < 0.5:
        max_rounds += 1
        print(f"问题解决进度不足，延长交互轮次。当前进度: {progress:.2f}")
    
    # 如果分析中包含工具调用，则使用它
    if "tool_calls" in parsed_analysis and len(parsed_analysis["tool_calls"]) > 0:
        next_tool_call = parsed_analysis["tool_calls"][0]
    else:
        # 根据当前进度决定是否需要继续搜索
        if progress < 0.7:
            # 进度不足，生成新的搜索查询
            search_terms = self._generate_follow_up_search(question, trace["steps"])
            next_tool_call = {
                "name": "brightdata_search" if random.random() < 0.7 else "fetch_content",
                "input": {
                    "content": search_terms
                }
            }
        else:
            # 进度较高，可能不需要更多搜索
            # 但仍有一定概率继续搜索以提高完整性
            if random.random() < 0.3:
                search_terms = self._generate_follow_up_search(question, trace["steps"])
                next_tool_call = {
                    "name": "brightdata_search" if random.random() < 0.5 else "fetch_content",
                    "input": {
                        "content": search_terms
                    }
                }
            else:
                # 决定不再搜索，直接进入最终答案阶段
                break
    
    # 执行工具调用和分析...
    
    # 增加轮次计数
    current_round += 1
```

## 4. 实施路径

### 4.1 第一阶段：基础框架搭建（1-2周）

1. 实现 `SearchQualityEvaluator` 和 `SearchStrategyManager` 类
2. 增强 `_execute_tool_call` 方法的错误处理
3. 添加基本的辅助方法

### 4.2 第二阶段：核心功能实现（2-3周）

1. 实现 `ProgressEvaluator` 类
2. 修改 `generate_trace` 方法中的交互轮次决策逻辑
3. 完善搜索策略和错误恢复机制

### 4.3 第三阶段：测试和优化（1-2周）

1. 使用现有测试问题集进行测试
2. 分析轨迹质量和成功率
3. 优化参数和阈值

## 5. 后续扩展

完成上述实施后，可以考虑以下扩展：

1. 集成基础Python数据分析和可视化能力
2. 扩展内容获取器，支持更多网站和内容类型
3. 实现更复杂的问题分解和推理能力
