# Copyright 2025 xDAN Agentic RL ToolRL Team
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

import json
import os
import re
from collections import Counter

import numpy as np

STEP_BOUNDARY = 30


def call_llm_as_a_judge(prompt):
    from openai import OpenAI

    # Use OpenAI API directly
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key="EMPTY",
        base_url="http://10.110.10.3:8000/v1",
    )

    models = client.models.list()
    model = models.data[0].id

    json_schema = {"type": "object", "properties": {"reward": {"type": "number", "minimum": -1, "maximum": 1}}, "required": ["reward"]}

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"guided_json": json_schema},
    )
    score = completion.choices[0].message.content
    return score


PROMPT = """You will be given a response and its ground truth. You need to judge if the response correctly reflected all the aspects based on the ground truth.
Your score for the judgement should be a float number between -1 and 1, where 1 means the pred_answer is completely correct to ground truth, and -1 means the
pred_answer is completely incorrect to ground truth

Here is the criteria for the judgement:
1. The response doesn't need to be exactly the same as any of the ground truth, but should be semantically same.

response: {pred_answer}
ground truth: {gt_answer}

The output should in the following json format:
```json
{{
"reward": score
}}
```
Your output:
"""


def cosine_similarity(a, b):
    """Calculate the cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    result = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    # Convert to float to avoid numpy type issues
    return float(result) if not np.isnan(result) else 0.0


def preprocess_text(text: str) -> str:
    """预处理文本，用于数据集的评分

    处理步骤:
    1. 转换为小写
    # 2. 移除标点符号 (.,!?;:'"()[]{}...)
    3. 去除多余空格
    """
    # # 将标点符号替换为空格
    # for punct in string.punctuation:
    #     text = text.replace(punct, " ")

    # 替换多个空格为单个空格
    text = re.sub(r"\s+", " ", text)

    # 去除首尾空格
    text = text.strip()
    return text


def compute_response_score(solution_str, ground_truth, tokenizer, reward_type, val_type="f1") -> float:
    solution_str = solution_str.lower()
    ground_truth = ground_truth.split("<response>")[1].split("</response>")[0].lower().strip()
    ground_truths = [ground_truth]
    # 使用正则提取第一个<response>标签中的内容
    try:
        answer_match = re.search(r"<response>(.*?)</response>", solution_str, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            # 对答案进行预处理
            answer_content = preprocess_text(answer_content)
        else:
            return 0.0  # 如果没有response标签，表示格式错误, 返回0.0（格式错误另外惩罚）
    except Exception as e:
        print(f"Error extracting answer content: {e}")
        return 0.0

    max_score = 0.0

    for gt in ground_truths:
        # 对ground truth进行预处理
        gt = preprocess_text(gt)

        try:
            if reward_type == "embedding":
                from infinity_client import Client
                from infinity_client.api.default import embeddings
                from infinity_client.models import OpenAIEmbeddingInputText, OpenAIEmbeddingResult

                with Client(base_url="http://10.110.10.3:3000") as client:
                    embeds: OpenAIEmbeddingResult = embeddings.sync(
                        client=client,
                        body=OpenAIEmbeddingInputText.from_dict(
                            {
                                "input": [gt, answer_content],
                                "model": "jina-embeddings-v2-base-zh",
                            }
                        ),
                    )

                    print("Use embedding model scoring!")
                    return cosine_similarity(embeds.data[0].embedding, embeds.data[1].embedding)
            else:
                prompt = PROMPT.format(pred_answer=answer_content, gt_answer=gt)
                score = json.loads(call_llm_as_a_judge(prompt))["reward"]
                print("LLM as a judger", score)
                return float(score)
        except Exception as e:
            print(f"Error during LLM scoring: {e}")
            print("Using fallback scoring...")
            if val_type == "em":
                print("Using exact match scoring...")
                if gt == answer_content:
                    return 1.0
            elif val_type == "f1":
                print("Using f1 scoring...")

                # 将答案和参考答案分词
                pred_tokens = set(tokenizer.encode(answer_content))
                gt_tokens = set(tokenizer.encode(gt))
                print("pred_tokens:", [tokenizer.decode([token]) for token in pred_tokens])
                print("gt_tokens:", [tokenizer.decode([token]) for token in gt_tokens])

                if not gt_tokens:  # 避免除零错误
                    continue
                if not pred_tokens:
                    continue

                # 计算共同的词数
                common_tokens = pred_tokens & gt_tokens

                # 计算精确率和召回率
                precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
                recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0

                # 计算F1分数
                if precision + recall > 0:  # 避免除零错误
                    f1 = 2 * (precision * recall) / (precision + recall)
                    max_score = max(max_score, f1)
    return max_score


def match_score(list1, list2):
    """Compute a similarity score considering element frequency, ignoring order."""
    if list1 == list2:
        return 1.0

    if os.getenv("REFINEDREWARD", 0) == "1":
        print("REFINEDREWARD is set to 1, so strict match is used")
        if list1 != list2:
            return 0.0

    if not list1 or not list2:
        return 0.0

    count1 = Counter(list1)  # Frequency count for list1
    count2 = Counter(list2)  # Frequency count for list2

    intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
    max_possible = len(list1) + len(list2) - intersection

    return intersection / max_possible if max_possible > 0 else 0.0


# custoimzed reward functions: format
def customize_format_reward_func(completions, answer, step, max_possible_reward, min_possible_reward, reward_type, **kwargs):
    if str(os.getenv("MAX1STEP30MAX3", 0)) == "1":
        print("MAX1STEP30MAX3 is set to 1, so max 1 -> 30 steps -> max 3")
        if step >= STEP_BOUNDARY:
            max_possible_reward = max_possible_reward / 2
            min_possible_reward = min_possible_reward / 2
        else:
            max_possible_reward = max_possible_reward
            min_possible_reward = min_possible_reward

    # schedule reward
    if str(os.getenv("SCHEDULEREWARD", 0)) == "1":
        print("SCHEDULEREWARD is set to 1, so schedule reward is used")
        max_possible_reward = 2 - (2 - max_possible_reward) * step / 150
        min_possible_reward = -2 + (2 + min_possible_reward) * step / 150
        if max_possible_reward < 1.0:
            max_possible_reward = 1.0
        if min_possible_reward > -1.0:
            min_possible_reward = -1.0

    rewards = []
    responses = [completion[0]["content"] for completion in completions]

    print("\n======= Answer ======= ")
    print(answer[0])
    print("\n======= Responses ======= ")
    for idx, response in enumerate(responses):
        print(f"*** Response {idx + 1}***\n{response}")

    for response in responses:
        response = response.strip()
        reward = min_possible_reward
        if reward_type == "tool_use":
            pattern = r"^<think>.*?</think>\n*<tool_call>.*?</tool_call>$"
            if re.search(pattern, response, re.DOTALL) and response.count("<tool_call>") == 1 and response.count("</tool_call>") == 1:
                reward = max_possible_reward
        else:
            pattern = r"^<think>.*?</think>\n*<response>.*?</response>$"
            if re.search(pattern, response, re.DOTALL) and response.count("<response>") == 1 and response.count("</response>") == 1:
                reward = max_possible_reward

        rewards.append(reward)

    print("\n======= Reward for <format> =======")
    print("Reward function for <format> is called ...")
    print(rewards)
    return rewards


# customized reward functions: length
def customize_length_reward_func(completions, answer, step, max_possible_reward, min_possible_reward, **kwargs):
    # schedule length
    if os.getenv("SCHEDULELENGTH", 0) == "1":
        print("SCHEDULELENGTH is set to 1, so schedule max reward for length is used")
        max_reward_len = (640 - 384) * step / 105 + 384
    else:
        max_reward_len = 512

    """Reward function that gives higher scores to longer completions."""
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for response, ans in zip(responses, answer):
        if "<think>" not in response or "</think>" not in response:
            rewards.append(min_possible_reward)
            continue
        think_responses = response.split("<think>")[-1].split("</think>")[0].strip()
        reward = round(len(think_responses.split()) / max_reward_len, 2)
        if reward > 1.0:
            reward = 1.0

        final_reward = reward * (max_possible_reward - min_possible_reward) + min_possible_reward
        rewards.append(final_reward)

    print("\n======= Reward for <length> =======")
    print("Reward function for <length> is called ...")
    print(rewards)
    return rewards


def compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward, min_possible_reward):
    if gt_tools == pd_tools:
        print("Max possible score:", "Exact Match!")
        print("Score:", max_possible_reward)
        return max_possible_reward

    if os.getenv("COARSEREWARD", 0) == "1":
        print("COARSEREWARD is set to 1, so coarse reward is used")
        if gt_tools != pd_tools:
            return min_possible_reward

    gt_names = [tool["name"] for tool in gt_tools]
    pd_names = [tool["name"] for tool in pd_tools]
    score = match_score(list(gt_names), list(pd_names))

    local_max_possible = 1.0
    used_pd_indices = set()  # Keep track of matched pd_tools

    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool["parameters"]

        if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
            print("INTERMEDIATEREWARD is set to 1, so local max possible is changed")
            local_max_possible += 1.0
        else:
            local_max_possible += 1.0 + len(gt_params)

        best_match = None
        best_match_score = 0.0
        best_match_index = -1

        # Find the best matching unused pd_tool
        for i, pd_tool in enumerate(pd_tools):
            if i in used_pd_indices or pd_tool["name"] != gt_name:
                continue

            if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
                if gt_tool == pd_tool:
                    best_match = pd_tool
                    best_match_index = i
                    best_match_score = 1.0
                    break
                else:
                    continue

            pd_params = pd_tool["parameters"]
            param_score = match_score(list(gt_params.keys()), list(pd_params.keys()))

            # Calculate correctness score for parameter values
            correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)

            total_score = param_score + correctness_score

            if total_score > best_match_score:
                best_match_score = total_score
                best_match = pd_tool
                best_match_index = i

        if best_match:
            used_pd_indices.add(best_match_index)
            score += best_match_score

    print()
    print("Max possible score:", local_max_possible)
    print("Score:", score)

    return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward


# custoimzed reward functions: tool call correctness
def customize_correctness_reward_tool(completions, answer, step, max_possible_reward, min_possible_reward, tokenizer, reward_type, **kwargs):
    if str(os.getenv("MAX1STEP30MAX3", 0)) == "1":
        print("MAX1STEP30MAX3 is set to 1, so max 1 -> 30 steps -> max 3")
        if step < 30:
            max_possible_reward = max_possible_reward / 3
            min_possible_reward = min_possible_reward / 3
        else:
            max_possible_reward = max_possible_reward
            min_possible_reward = min_possible_reward

    if str(os.getenv("SCHEDULEREWARD", 0)) == "1":
        print("SCHEDULEREWARD is set to 1, so schedule reward is used")
        max_possible_reward = (max_possible_reward - 2) * step / 150 + 2
        min_possible_reward = (min_possible_reward + 2) * step / 150 - 2
        if max_possible_reward > 3.0:
            max_possible_reward = 3.0
        if min_possible_reward < -3.0:
            min_possible_reward = -3.0

    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for response, ans in zip(responses, answer):
        reward = 0.0

        if reward_type == "tool_use":
            if "<tool_call>" not in ans:
                # if "<tool_call>" not in response and "</tool_call>" not in response:
                #     reward = max_possible_reward
                # else:
                #     reward = min_possible_reward
                rewards.append(reward)
                continue

            gt_tool_call = ans.split("<tool_call>")[1].split("</tool_call>")[0].strip()
            gt_tools = gt_tool_call.split("\n")
            # each diction contains "name" and "parameter"
            gt_tools = [json.loads(tool) for tool in gt_tools]

            try:
                # Change here as a constrint in training: if the format is not correct, directly give the lowest possible score
                assert "<tool_call>" in response
                assert "</tool_call>" in response
                pd_tools = response.split("<tool_call>")[1].split("</tool_call>")[0].strip().split("\n")
                pd_tools = [json.loads(tool) for tool in pd_tools]
                reward = compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward, min_possible_reward)  # top reward is 2
            except Exception:
                reward = min_possible_reward
        else:
            reward = compute_response_score(response, ans, tokenizer, reward_type) * max_possible_reward

        rewards.append(reward)

    print("\n======= Reward for <tool call> =======")
    print("Reward function for <tool call> correctness is called ...")
    print(rewards)
    return rewards


def compute_score(solution_str, ground_truth, extra_info, tokenizer, step=0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    exp_name = str(os.getenv("EXPERIMENT_NAME", ""))
    if "llama" in exp_name:
        predict_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
    elif "qwen" in exp_name:
        predict_str = solution_str.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    else:
        raise NotImplementedError(f"Unknown model name: {exp_name}")

    if str(os.getenv("CORRECTMAX1", 0)) == "1":
        print("CORRECTMAX1 is set to 1, so max score is set to 1")
        tool_max_possible = 1.0
        tool_min_possible = -1.0
    else:
        tool_max_possible = 3.0
        tool_min_possible = -3.0

    format_max_possible = 1.0
    format_min_possible = 0.0

    length_max_possible = 1.0
    length_min_possible = 0.0

    completions = [[{"role": "assistant", "content": predict_str}]]
    answer = [ground_truth]

    print(extra_info)
    format_score = customize_format_reward_func(completions, answer, step, format_max_possible, format_min_possible, extra_info["type"])[0]
    correctness_score = customize_correctness_reward_tool(completions, answer, step, tool_max_possible, tool_min_possible, tokenizer, extra_info["type"])[0]

    if str(os.getenv("WITHLENGTH", 0)) == "1":
        print("WITHLENGTH is set to 1, so length score is set!")
        length_score = customize_length_reward_func(completions, answer, step, length_max_possible, length_min_possible)[0]
    else:
        length_score = 0

    score = format_score + correctness_score + length_score

    return {"score": score, "format_score": format_score, "correctness_score": correctness_score, "length_score": length_score}
