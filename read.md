# ToolRL: Reward is All Tool Learning Needs

**作者：** Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek Hakkani-Tür, Gokhan Tur, Heng Ji  
**机构：** University of Illinois Urbana-Champaign  
**联系方式：** {chengq9, hengji}@illinois.edu

## 摘要

Current Large Language Models (LLMs) often undergo supervised fine-tuning (SFT) to acquire tool use capabilities. However, SFT struggles to generalize to unfamiliar or complex tool use scenarios. Recent advancements in reinforcement learning (RL), particularly with R1-like models, have demonstrated promising reasoning and generalization abilities. Yet, reward design for tool use presents unique challenges: multiple tools may be invoked with diverse parameters, and coarse-grained reward signals, such as answer matching, fail to offer the fine-grained feedback required for effective learning.

In this work, we present the first comprehensive study on reward design for tool selection and application tasks within the RL paradigm. We systematically explore a wide range of reward strategies, analyzing their types, scales, granularity, and temporal dynamics. Building on these insights, we propose a principled reward design tailored for tool use tasks and apply it to train LLMs using Group Relative Policy Optimization (GRPO). Empirical evaluations across diverse benchmarks demonstrate that our approach yields robust, scalable, and stable training, achieving a 17% improvement over base models and a 15% gain over SFT models. These results highlight the critical role of thoughtful reward design in enhancing the tool use capabilities and generalization performance of LLMs. All the code are released to facilitate future research.¹

## 1. 引言

Recent advances in Large Language Models (LLMs) have showcased remarkable capabilities in complex reasoning tasks (Kumar et al., 2025). Among the techniques that have significantly contributed to this progress, Reinforcement Learning (RL) has emerged as a powerful paradigm, enabling

¹ Data and codes released at https://github.com/qiancheng0/ToolRL
Figure 1: SFT on distilled deep-thinking trajectories
suffers from overthinking and limited generalization.
LLMs to develop emergent capabilities such as selfreflection, self-correction, and long-horizon planning (Guo et al., 2025; Team et al., 2025). These
capabilities have been instrumental in the success
of models like o1 and R1, particularly in mathematical and logical reasoning domains (Qin et al.,
2024a; Huang et al., 2024; Li et al., 2025b; Kang
et al., 2025).
Beyond traditional reasoning tasks, an increasingly important area is Tool-Integrated Reasoning
(TIR). TIR involves LLMs interacting with external tools, such as search engines (Jin et al., 2025;
Zheng et al., 2025), calculators (Chen et al., 2023b;
Qin et al., 2023), or code interpreters (Gou et al.,
2023; Liao et al., 2024), in a multi-step, feedbackdriven loop to arrive at solutions. TIR is particularly important because it addresses core limitations of LLMs, such as outdated knowledge, calculation inaccuracy, and shallow reasoning. By
integrating external tools that offer real-time access
and specialized capabilities, TIR enables models
to tackle complex tasks in a more grounded and
goal-directed way.
Unlike textual reasoning, which primarily involves deduction and inference from static text,
arXiv:2504.13958v1 [cs.LG] 16 Apr 2025
20
30
40
50
60
Accuracy (%)
46.20%
52.98%
58.38%
44.10%
BFCL Benchmark
20
30
40
50
60
70
Accuracy (%)
44.00%
60.00%
72.00%
52.00%
Bamboogle Benchmark
Qwen2.5-1.5B Qwen2.5-3B Qwen2.5-7B LLaMA3.2-3B
30
35
40
45
50
55
60
65
70
Accuracy (%)
63.15%
67.00%
64.66%
59.13%
API-Bank Benchmark
Raw SFT400 SFT4K SFT400+PPO SFT400+GRPO PPO Cold Start GRPO Cold Start
0 20 40 60 80
Step
0.50
0.25
0.00
0.25
0.50
0.75
1.00
1.25
Mean Format Reward
Qwen2.5-7B
Qwen2.5-1.5B
LLaMA3.2-3B
Qwen2.5-3B
0 20 40 60 80
Step
4
2
0
2
4
Mean Correctness Reward
Qwen2.5-7B
Qwen2.5-1.5B
LLaMA3.2-3B
Qwen2.5-3B
Figure 2: Main results (left) and reward trends over training steps for GRPO Cold Start across four models (right).
GRPO Cold Start, equipped with our proposed reward design, consistently achieves the highest performance, with
reward curves showing a rapid increase during training.
TIR additionally demands the model’s ability to
select appropriate tools, interpret intermediate outputs, and adaptively refine its trajectory on the fly.
These dynamic and interactive reasoning skills position TIR at the core of the emerging paradigm
of LLMs-as-agents. As such, TIR enables a wide
range of applications, including scientific discovery (Roohani et al., 2024; Inoue et al., 2024), research automation (Baek et al., 2024; Wang et al.,
2024), embodied task completion (Zhang et al.,
2023; Huang et al., 2023), and everyday decisionmaking (Ye et al., 2023; Zhai et al., 2024).
Training LLMs for TIR tasks has predominantly
relied on Supervised Fine-Tuning (SFT), wherein
existing approaches typically generate these integrated reasoning steps offline, followed by subsequent SFT on these trajectories (Chen et al., 2023a;
Zeng et al., 2024; Chen et al., 2024; Acikgoz et al.,
2025). While SFT is effective to some extent,
it struggles with generalization, exploration, and
adaptability (Chu et al., 2025; Guo et al., 2025). As
illustrated in Figure 1, a model trained with SFT on
deep-thinking trajectories over-interprets the tool
and fails to reject the inappropriate tool, merely
imitating cues like “but wait” without engaging in
genuine deep thinking. As such, SFT often fails to
capture the strategic flexibility needed for optimal
tool use, particularly in open-ended or multi-step
settings. This motivates a fundamental research
question: Can RL-based training methods better
equip LLMs with agentic tool-using capabilities,
and if so, what is the optimal RL design for TIR?
Recent efforts such as Search-R1 (Jin et al.,
2025) and TORL (Li et al., 2025b) have begun to
explore this direction. However, their focus is narrow: either constrained to search tools in question
answering settings or code tools in math problemsolving. In contrast, our work aims to study RLbased training for general-purpose tool selection
and application, across diverse and complex tool
sets with different task types.
For an RL algorithm to be effective, a welldesigned reward is essential. Unlike math tasks
with a single correct answer, Tool-Integrated Reasoning (TIR) tasks introduce multiple layers of
complexity: they often involve multi-step interactions where each turn may require invoking multiple tools, each with carefully specified parameters. Designing effective reward signals to guide
learning through this complexity remains an open
and underexplored challenge. In this paper, we
focus on the problem of reward design for TIR
and propose a principled, generalizable framework
that can be applied across various RL algorithms.
While our reward design is algorithm-agnostic by
nature, we empirically demonstrate its effectiveness using both Group Relative Policy Optimization (GRPO) (Shao et al., 2024) and Proximal Policy Optimization (PPO) (Schulman et al., 2017),
showcasing its versatility and impact on improving
tool use performance.
We begin by formalizing the TIR task, and out-
lining general principles for effective reward design. Building on this foundation, we show how
RL algorithm can be leveraged to train LLMs for
robust and context-aware tool selection and application. Empirical results demonstrate that our approach outperforms base models by 17% and SFT
models by 15% across multiple tool use and QA
benchmarks. Moreover, the trained model exhibits
strong generalization to unseen scenarios and task
objectives, along with emergent behaviors such as
proactiveness and metacognitive reasoning.
To identify optimal reward strategies, we next
systematically explore a broad spectrum of reward
configurations across four key dimensions: (1) reward type (what aspect to reward), (2) reward scale
(how much to reward), (3) reward granularity (how
detailed the reward signal is), and (4) reward dynamics (how rewards evolve over time). Through
extensive experiments, we identify reward designs
that best align with agentic tool use and uncover
insights into what makes a reward “useful” for tool
invoking LLMs. We summarize the core insights
we derive as follows:
• Longer reasoning trace is not inherently better
and length rewards can degrade performance.
• Dynamic reward scale helps models transition
smoothly from simple to complex behaviors.
• Finegrained reward decomposition leads to more
stable and effective learning.
We also summarize the overall contributions of our
paper as follows:
• We present the first systematic study on RLbased training for general-purpose tool selection
and application in LLMs.
• We propose a principled reward design framework tailored for TIR and validate its effectiveness through RL algorithms including GRPO.
• We conduct extensive experiments analyzing the
effects of various reward strategies and distill
actionable insights for future research on LLMagent training.
This work pioneers the application of RL to general
TIR and provides the first empirical roadmap for
reward design in TIR, paving the way toward more
capable and autonomous LLM agents.
2 Related Work
Tool-Integrated Reasoning of LLMs. Toolintegrated reasoning (TIR) has emerged as a
promising approach to enhance the capabilities of
LLMs. Early studies introduced the concept of
equipping LLMs with external tools to overcome
their inherent limitations (Schick et al., 2023; Qin
et al., 2024b; Yao et al., 2023), such as program
executors (Chen et al., 2022) and search engines
(Vu et al., 2023). To systematically assess these enhanced capabilities, several benchmarks have been
proposed to evaluate tool use performance across
various dimensions, including API selection, argument generation, and generalization (Qin et al.,
2024c; Patil et al., 2023; Qian et al., 2024a). Building on this foundation, subsequent research has focused on constructing high-quality tool use datasets
(Liu et al., 2024; Qian et al., 2025), enabling models to autonomously create and invoke tools (Qian
et al., 2023, 2024b), and applying these techniques
to problems spanning different modalities (Shen
et al., 2025) and specialized domains (Ling et al.,
2023). More recently, reinforcement learning (RL)
has been explored as an effective framework to further improve TIR, demonstrating success in tasks
such as information retrieval (Jin et al., 2025) and
math computation (Li et al., 2025b). These advances collectively highlight the growing potential
of tool-augmented LLMs for general-purpose reasoning in open-domain settings.
Exploration of RL in LLMs. Previous work has
primarily relied on supervised fine-tuning (SFT)
with carefully curated datasets to enhance LLM
performance in tool use (Schick et al., 2023; Qin
et al., 2024c). Recently, reinforcement learning (RL) has gained traction as a more scalable
and generalizable training paradigm. The development of RL methods for LLMs has evolved
from reinforcement learning from human feedback
(RLHF) (Kaufmann et al., 2023) and proximal policy optimization (PPO) (Schulman et al., 2017) to
more advanced techniques such as direct preference optimization (DPO) (Rafailov et al., 2023),
SimPO (Meng et al., 2024), and group relative
policy optimization (GRPO) (Shao et al., 2024).
Extensions like dynamic sampling policy optimization (DAPO) (Yu et al., 2025) and the more recent
value-based augmented proximal policy optimization (VAPO) (Yuan et al., 2025) further improve
training stability and efficiency.
Among these, GRPO (Shao et al., 2024) is specifically designed for LLMs, replacing the traditional
critic with a group-based evaluation strategy. It has
shown strong performance in enhancing reasoning
abilities across a range of tasks, including math-
ematical problem solving (Shao et al., 2024; Xie
et al., 2025), search engine interaction (Jin et al.,
2025; Song et al., 2025), and code generation (Li
et al., 2025b). Beyond task variety, recent studies
have analyzed the influence of dataset scale (Li
et al., 2025a) and GRPO’s effectiveness in smaller
model settings (Dang and Ngo, 2025). GRPO’s
flexible reward function enables adaptation to diverse objectives, such as assigning weights to subtasks (Yu et al., 2024) or constraining tool use frequency (Li et al., 2025b). In this work, we extend
GRPO to enhance general tool use capabilities, improving LLMs’ ability to select and interact with
external tools across a wide range of scenarios.
3 Method
Supervised fine-tuning (SFT), as illustrated in Figure 1, often suffers from overfitting to certain patterns and constrains the model’s ability to learn
optimal strategies for tool use. To address this, we
introduce a reinforcement learning (RL) approach
for enhancing tool-integrated reasoning (TIR) in
LLMs. In this section, we begin by defining the
TIR task (Section 3.1), followed by our customized
rollout strategy (Section 3.2) and reward design
(Section 3.3). These components are then integrated into the Group Relative Policy Optimization
(GRPO) framework (Shao et al., 2024) to guide
model training on general TIR tasks (Section 3.4).
3.1 Task Definition
Tool-Integrated Reasoning (TIR) is the process of
incorporating external tools into the reasoning trajectory of an LLM to solve a user task. A typical
TIR trajectory involves multiple tool invocations
over several reasoning steps, with the final outcome
determined by the cumulative success of these intermediate decisions.
Formally, given a tool set T = {t1, t2, . . . , tn}
containing n available tools, and a user query Q,
the reasoning trajectory up to step k is denoted as:
sk = (r1, T1, o1),(r2, T2, o2), . . . ,(rk, Tk, ok),
where ri denotes the model’s natural language reasoning at step i, Ti ⊆ T denotes the set of tool
calls invoked at step i, and oi denotes the observation received after executing tools in Ti
, possibly
including both environment and user feedback.
At each step k + 1, the model must generate
the next reasoning step rk+1, select a set of tools
Tk+1 ⊆ T , and formulate a grounded tool call (i.e.,
a parameterized invocation of each tool) to make
progress toward solving Q.
The model’s policy is defined as π : sk →
(rk+1, Tk+1), where the model’s objective at each
step is to select a tool set Tk+1 that maximizes the
immediate reward:
T
∗
k+1 = arg max
Tk+1⊆T
R(sk, Tk+1, ok+1),
where R(·) represents the reward function that evaluates progress made by invoking the tools in Tk+1.
While the immediate reward at each step is maximized, the model’s policy is implicitly optimized
to maximize the cumulative reward over the entire
trajectory, formulated as:
max
π
Eπ
"X
K
k=1
R(sk, Tk+1, ok+1)
#
,
This formulation is valid because our training data
includes ground truth tool calls at each step, allowing step-wise reward signals to guide multi-step
success. Unlike QA tasks that focus solely on the
final answer, tool selection and application tasks
provide dense intermediate feedback. Moreover,
we later demonstrate that our method enables the
model to generalize to settings where tool calls
are free-form and only the final outcome matters.
Therefore, out task setting encourages the model to
optimize tool use at each step while aligning with
the overall task goal.
3.2 TIR Rollout
To enable the model to autonomously generate reasoning traces and tool calls, we utilize a system
prompt as shown in Figure 4 during rollout. The
Tool List placeholder denotes the tool set T , which
contains all tools available for invocation. We indicate in the instruction that the LLM should use special tokens <think>, <tool_call>, and <response>
to indicates their thoughts, tool calls and responses
in output.
As illustrated in Figure 3, when the model output includes <tool_call>, we automatically parse
the tool calls into individual invocations using the
model-predicted parameters. The outputs from executions are then inserted into the <obs> field and
appended to the dialogue history, whose format is
shown in Figure 12, serving as the model’s interaction trajectory. Similarly, if the output contains
<response>, the corresponding response is parsed
and appended to the dialogue history.
Figure 3: Illustration of TIR rollout and calculation of format and correctness reward.
It is important to note that <tool_call> and <response> are not mutually exclusive; they may cooccur within a single output. The user’s initial
query Q is placed in the Initial User Input placeholder, and any subsequent user inputs are also
appended to the dialogue history when present.
3.3 Reward Design
Rule-based reward mechanisms have demonstrated
strong empirical performance and are commonly
employed. In our training, we similarly adopt a
reward formulation that combines structural and
correctness-based components, in line with prior
works (Jin et al., 2025; Li et al., 2025b; Xie et al.,
2025). Specifically, the format reward assesses
whether the model output adheres to the expected
structure including thoughts, tool calls, and responses, while the correctness reward evaluates the
accuracy of tool invocations. Formally, the overall
reward Rfinal(·) is decomposed into two components: Rformat + Rcorrect, each described in detail
below:
Format Reward. The format reward Rformat ∈
{0, 1} checks whether the model output contains
all required special tokens in the correct order as
specified by the ground truth:
Rformat =



1,
if all required fields appear
and are in the correct order
0, otherwise
Correctness Reward. The correctness reward
Rcorrect ∈ [−3, 3] evaluates predicted tool calls
P = {P1, ..., Pm} against ground-truth calls G =
{G1, ..., Gn}. It includes three components:
• Tool Name Matching:
rname =
|NG ∩ NP |
|NG ∪ NP |
∈ [0, 1]
where NG and NP are the sets of tool names
extracted from the ground-truth and predicted
tool calls, respectively.
• Parameter Name Matching:
rparam =
X
Gj∈G
|keys(PG) ∩ keys(PP )|
|keys(PG) ∪ keys(PP )|
∈ [0, |G|]
where keys(PG) and keys(PP ) represent the parameter names of the predicted and ground-truth
tool calls, respectively.
• Parameter Content Matching:
rvalue =
X
Gj∈G
X
k∈keys(Gj )
1[PG[k] = PP [k]]
∈ [0,
X
Gj∈G
|keys(Gj )|]
where PG[k]] and PP [k] represent the values of
the parameters for the predicted and ground truth
tool calls.
• Total match score for each match is:
rmatch = rname + rparam + rvalue ∈ [0, Smax]
where Smax = 1 + |G| +
P
Gj∈G |keys(Gj )|
denotes the maximum possible score.
The total score is computed by finding the optimal
matching between P and G to maximize the total
match score:
Rcorrect = 6 ·
Rmax
Smax
− 3 ∈ [−3, 3]
where Rmax denotes the total match score from
the optimal matching. The final correctness reward
System Prompt for Training
You are a helpful dialogue assistant capable of leveraging tool calls to solve user tasks and provide
structured chat responses.
Available Tools
In your response, you can use the following tools:
{{Tool List}}
Steps for Each Turn
1. Think: Recall relevant context and analyze the current user goal.
2. Decide on Tool Usage: If a tool is needed, specify the tool and its parameters.
3. Respond Appropriately: If a response is needed, generate one while maintaining consistency
across user queries.
Output Format
<think> Your thoughts and reasoning </think>
<tool_call>
{“name”: “Tool name”, “parameters”: {“Parameter name”: “Parameter content”, “... ...”: “... ...”}}
{“name”: “... ...”, “parameters”: {“... ...”: “... ...”, “... ...”: “... ...”}}
...
</tool_call>
<response> AI’s final response </response>
Important Notes
1. You must always include the <think> field to outline your reasoning. Provide at least one
of <tool_call> or <response>. Decide whether to use <tool_call> (possibly multiple times),
<response>, or both.
2. You can invoke multiple tool calls simultaneously in the <tool_call> fields. Each tool call
should be a JSON object with a “name” field and a “parameters” field containing a dictionary of
parameters. If no parameters are needed, leave the “parameters” field an empty dictionary.
3. Refer to the previous dialogue records in the history, including the user’s queries, previous
<tool_call>, <response>, and any tool feedback noted as <obs> (if exists).
Figure 4: The system prompt used for TIR’s rollout.
Rcorrect is the normalized reward for the matching
process. We empirically set the reward scale within
the range of [−3, 3], with more analysis and ablatiions of reward scale presented in Section 5.
The final reward value Rfinal is finally derived as
the sum of Rformat and Rcorrect:
Rfinal = Rformat + Rcorrect ∈ [−3, 4]
Unlike prior works that often rely on binary or
overly simplified reward signals, our design captures the nuanced structure of tool calls by evaluating multiple interdependent components including
tool names, parameter schemas, and parameter values. This fine-grained formulation better reflects
the complexity of real-world tool use, where correctness cannot be reduced to a single binary criterion. We further validate the impact of this design
through comprehensive analysis in Section 5.
Overall, our reward design ensures a balanced
and interpretable evaluation signal by explicitly
separating structural compliance from semantic
correctness. By aligning rewards with both format adherence and fine-grained tool call accuracy,
the model is guided to produce outputs that are not
only syntactically valid but also semantically faithful, which is crucial for downstream tool execution
and final task success.
3.4 RL Training with GRPO
To tune the model with structured rewards, we employ GRPO, a variant of PPO that introduces advantage normalization within grouped samples. This
normalization helps stabilize training by reducing
variance across samples that share a common input
context. Let πθ represent the current policy.
Normalized Advantage Across Query Groups.
For each query Q, its responses derived from the
rollout form a group GQ consisting of multiple
responses and their corresponding reward values:
GQ = {A,(s1, r1),(s2, r2), . . . ,(sn, rn)}
where A denotes the ground-truth annotation for Q,
and each reward ri
is computed as the sum of the
format and correctness rewards associated with response si
, i.e., ri = Rformat(si
, A)+Rcorrect(si
, A).
For each group, we calculate the mean and standard
deviation of the rewards:
µQ =
1
n
Xn
i=1
ri, σQ =
vuut
1
n
Xn
i=1
(ri − µQ)
2
Then, for each sample si
in the group, we define
the normalized advantage:
Ai(si|Q) = ri − µQ
σQ + η
where η is a constant to avoid division by zero.
Policy Optimization Objective. The policy πθ
is optimized using the standard clipped PPO objective, adapted with our group-wise normalized
advantages:
JGRPO(θ) = EQ∼DEsi∼πθ
h
min 
πθ(si|Q)
πold(si|Q)
Ai(si|Q),
clip