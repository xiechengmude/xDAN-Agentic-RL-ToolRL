import json

# from vllm import LLM, SamplingParams
from openai import OpenAI

# Configuration
# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://14.103.133.112:8002/v1"
# nohup vllm serve /data/xdan/models/xDAN-R2-Thinking-32b-Agent-RL-step150 --port 8002 \
# --served-model-name xDAN-R2-Thinking-32b-Agent-RL-step150 --max-model-len 32768 \
# --tensor-parallel-size 4 > vllm.xDAN-R2-Thinking-32b-Agent-RL-step150.log &

MODEL = "/data/xdan/models/xDAN-R2-Thinking-32b-Agent-RL-step150"
MAX_TOKENS = 16384
TEMPERATURE = 0.7

# Define system prompt
SYSTEM_PROMPT = (
    "You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks and provide structured chat responses.\n\n"
    "**Available Tools**\n"
    "In your response, you can use the following tools:\n{tools}\n\n"
    "**Steps for Each Turn**\n"
    "1. **Think:** Recall relevant context and analyze the current user goal.\n"
    "2. **Decide on Tool Usage:** If a tool is needed, specify the tool and its parameters.\n"
    "3. **Respond Appropriately:** If a response is needed, generate one while maintaining consistency across user queries.\n\n"
    "**Output Format**\n```plaintext\n"
    "<think> Your thoughts and reasoning </think>\n"
    "<tool_call>\n"
    '{{"name": "Tool name", "parameters": {{"Parameter name": "Parameter content", "... ...": "... ..."}}}}\n'
    '{{"name": "... ...", "parameters": {{"... ...": "... ...", "... ...": "... ..."}}}}\n'
    "...\n"
    "</tool_call>\n"
    "<response> AI's final response </response>\n"
    "```\n\n"
    "**Important Notes**\n"
    "1. You must always include the `<think>` field to outline your reasoning. Provide at least one of `<tool_call>` or `<response>`. "
    "Decide whether to use `<tool_call>` (possibly multiple times), `<response>`, or both.\n"
    '2. You can invoke multiple tool calls simultaneously in the `<tool_call>` fields. Each tool call should be a JSON object with a "name" '
    'field and an "parameters" field containing a dictionary of parameters. If no parameters are needed, leave the "parameters" field an empty dictionary.\n'
    "3. Refer to the previous dialogue records in the history, including the user's queries, previous `<tool_call>`, `<response>`, and any tool feedback "
    "noted as `<obs>` (if exists).\n"
)

# Helper to format tools list for the system prompt, should be MCP compatible


def format_tools(tools):
    lines = []
    for idx, t in enumerate(tools, 1):
        props = json.dumps(t["parameters"]["properties"])
        lines.append(f"{idx}. Name: {t['name']}\nDescription: {t['description']}\nParameters: {props}")
    return "\n".join(lines)


# Parsing function calls from the model output


def parse_final_response(output):
    if "<response>" in output:
        return output.split("<response>")[1].split("</response>")[0].strip()
    return None


def do_tool_calls(output):
    calls = []
    results = []
    if "<tool_call>" in output:
        body = output.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        for line in body.splitlines():
            try:
                call = json.loads(line)
                calls.append(call)
            except json.JSONDecodeError:
                continue
    # Send the tool call to the actual tool service, and get the result
    for call in calls:
        tool_name = call["name"]
        tool_params = call["parameters"]
        # TODO: Do the actual tool call here
        # # Simulate tool result for demonstration
        tool_response = "Helsinki today is 10 degree."
        result = {"name": tool_name, "parameters": tool_params, "content": tool_response}
        results.append(result)
        print(f"Tool call: {tool_name}, Parameters: {tool_params}, Result: {result}")
    return results


# Main inference function
def run_chat(llm, tools, messages, turn_type="multi_turn"):
    USER_PROMPT = "**Dialogue Records History**\n"

    for msg_idx, message in enumerate(messages):
        if message["role"] == "system":
            continue

        elif message["role"] == "user":
            if turn_type == "multi_turn":
                USER_PROMPT += f"<user> {message['content'].strip()}\nUse the one or more necessary tool calls to complete the task. You could "
                USER_PROMPT += "perform tool calls for multiple rounds so you can try and error. Please make a comprehensive plan about how to "
                USER_PROMPT += "achieve the goal step by step, and begin to call the tool step by step. If no tools apply or required parameters "
                USER_PROMPT += "are missing, please also directly state this in your response without tool calls. </user>\n"
            else:
                USER_PROMPT += f"<user> {message['content'].strip()}\nIf there's no appropriate tools to apply or required parameters are missing, "
                USER_PROMPT += "please directly inform me in your response without any tool call, or call the tool with the name as 'None'. Otherwise, "
                USER_PROMPT += "you should use one or more necessary tool calls to complete the given task in this turn. </user>\n"

        elif message["role"] == "tool":
            tool_name = message["name"].strip()
            tool_result = message["content"].strip()
            USER_PROMPT += f"<obs> You have made the tool call {tool_name}. Execution returns: {tool_result} </obs>\n"
            if msg_idx == len(messages) - 1:
                USER_PROMPT += "\n<user> If you think you have completed the current task, or the task cannot be finished, please respond directly without "
                USER_PROMPT += "additional tool calls. If you encounter an error during tool execution or the task remains unfinished, retry with the one or "
                USER_PROMPT += "more necessary tool calls according to your thought and plan until completion. Based on the tool execution feedback, reflect on "
                USER_PROMPT += "if understanding or selectioin of tool is wrong, what tool calling step is missing, and how to achieve the task goal from now on. </user>\n"

        elif message["role"] == "assistant":
            # we make the thought shorter and more concise in the history
            content = message["content"].strip()
            message["content"] = content
            USER_PROMPT += f"\n{message['content'].strip()}\n"

    USER_PROMPT = USER_PROMPT.strip()

    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT.format(tools=format_tools(tools))}<|im_end|>\n<|im_start|>user\n{USER_PROMPT}<|im_end|>\n<|im_start|>assistant\n"

    texts = []
    if llm is None:
        # Use OpenAI API directly
        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        models = client.models.list()
        model = models.data[0].id

        # Completion API
        stream = True
        completion = client.completions.create(model=model, prompt=prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, stream=stream)

        generated_text = ""
        print(f"Prompt: {prompt!r}, Generated text: \n\n", end="")
        if stream:
            for chunk in completion:
                for choice in chunk.choices:
                    for text in choice.text:
                        print(text, end="", flush=True)
                        generated_text += text
        else:
            for chunk in completion:
                generated_text += chunk.choices[0].text
                print(texts)
        texts.append(generated_text)
    else:
        # params = SamplingParams(temperature=TEMPERATURE, max_tokens=16384)
        params = None
        # Generate the response
        outputs = llm.generate([prompt], params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            texts.append(generated_text)
            print(f"Prompt: {prompt!r}, Generated text: \n\n{generated_text!r}")

    results = do_tool_calls(texts[0])
    return texts[0], results


# Example usage
if __name__ == "__main__":
    history = [{"role": "user", "content": "What's the weather today in Helsinki?"}]

    # example available tools description for DeepResearch use case.
    tools = [{"name": "search", "description": "Search over a knowledge base.", "parameters": {"properties": {"query": {"type": "string"}}}}, {"name": "summarize", "description": "Summarize a document or webpage given its URI.", "parameters": {"properties": {"uri": {"type": "string"}}}}]

    # llm = LLM(model=MODEL, tensor_parallel_size=4)
    llm = None

    output = ""
    while parse_final_response(output) is None:
        output, tool_calls = run_chat(llm, tools, history)
        history.append({"role": "assistant", "content": output})
        for call in tool_calls:
            call["role"] = "tool"
            history.append(call)

    print("\n\nFinal response:\n", parse_final_response(output))
