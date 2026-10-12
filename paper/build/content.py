# -*- coding: utf-8 -*-
# Paper content for "Adaptive LLM Model Routing Using Contextual Bandits with Feedback".
# Citations are written as {key1,key2} and numbered in order of first appearance.
# Section VI numbers come from results/gsm8k_pilot_models_4090/ (tables.md, results.json, sweeps/sweeps.md)
# and data/outcomes_gsm8k_models_4090.jsonl.

TITLE = "Adaptive LLM Model Routing Using Contextual Bandits with Feedback"

ABSTRACT = (
    "Applications that can call several large language models (LLMs) must decide which model should answer each "
    "query. Larger models tend to produce stronger answers at higher cost, whereas smaller models are cheaper but may "
    "fail on complex tasks. This paper formulates per-query model selection as a contextual-bandit problem and solves "
    "it with the Linear Upper Confidence Bound (LinUCB) algorithm. Each query is converted into a compact context "
    "vector; LinUCB scores every candidate model by combining an estimated reward with an uncertainty bonus, selects "
    "one model, and updates that model's statistics from the observed outcome. The reward combines normalized "
    "response quality, cost, and latency with explicit weights. The router is ordinary software and does not modify "
    "LLM weights. A pilot study routes 300 GSM8K math word problems (90 development, 210 held-out) among Qwen3-4B, "
    "Qwen3-8B, and Qwen3-14B, served in FP8 with vLLM on one RTX 4090, and compares LinUCB with small-only, large-only, "
    "random, and rule-based routing by full-information replay over five seeds. On the held-out split the three models "
    "reach 94.8%, 95.7%, and 95.2% accuracy, while cost per request rises from 1.42 to 4.71 units. LinUCB learns to "
    "send 98.2% of queries to the 4B model, matching its accuracy (94.8%) at 69% lower cost than large-only routing "
    "and a cumulative reward of 175.6 versus 139.9; it does not, however, beat the small-only and rule-based "
    "baselines (176.0), which are already near-optimal on this single-category workload."
)

KEYWORDS = ("Large language models, model routing, contextual bandits, LinUCB, exploration-exploitation, "
            "cost-aware inference, feedback-driven learning, LLM serving.")

# Each section: ("H1", text) | ("H2", text) | ("P", text) | ("EQ", text, n) | special blocks handled in build.py
BODY = [
("H1", "I. Introduction"),
("P", "Large language models are now deployed behind many applications, and an application frequently has access to "
      "more than one model. These models differ in response quality, inference latency, and computational or API cost. "
      "Sending every request to the largest model can produce strong answers but increases cost and delay for queries "
      "that a smaller model could answer; sending every request to the smallest model is cheaper but may fail on "
      "multi-step reasoning, code generation, or other complex tasks."),
("P", "Prior work trades cost against quality by choosing among models. LLM cascades query models in sequence and stop "
      "when an answer is judged acceptable {frugal}, while learned routers predict which model a query should be sent "
      "to {hybrid,routellm,shnitzer,zooter}, and RouterBench formalizes their evaluation {routerbench}. Most of these "
      "routers are trained offline and remain fixed after deployment. Contextual bandits instead learn from operation: "
      "a learner observes a context, chooses one action, and receives a reward only for that action {lattimore}. "
      "LinUCB models expected reward as a linear function of the context and adds an upper-confidence exploration "
      "bonus {lics,chu}. In LLM routing the context describes the query, the actions are candidate models, and the "
      "reward expresses the usefulness of the response relative to its cost and latency {mixllm,pilot}."),
("P", "This paper makes four contributions. First, it defines an end-to-end routing loop that combines interpretable "
      "query features, disjoint LinUCB, and an explicit quality-cost-latency reward with consistent treatment of "
      "failures. Second, it specifies a controlled same-family model pool and a full-information replay protocol that "
      "compares LinUCB with small-only, large-only, random, and rule-based routing. Third, it reports a pilot on 300 "
      "GSM8K problems with Qwen3-4B, 8B, and 14B on a single RTX 4090, including reward-weight, exploration, ablation, "
      "and bandit-variant sweeps. Fourth, it reports where LinUCB does not help: on this single-category workload it "
      "matches but does not exceed the best fixed policy. The router never modifies model weights; it learns which "
      "model to select for which kinds of queries. Five research questions guide the study: how to represent a query "
      "as a context (RQ1); whether LinUCB beats fixed, random, and rule-based routing under the same reward (RQ2); how "
      "reward weights move the operating point (RQ3); how quickly routing adapts to shift (RQ4); and in which task "
      "categories adaptive routing helps (RQ5)."),

("H1", "II. Related Work"),
("P", "FrugalGPT {frugal} reduces LLM API cost with a cascade that stops once a scorer judges a response reliable, and "
      "Hybrid LLM {hybrid} routes between a small and a large model by predicted difficulty. RouteLLM {routellm} learns "
      "a strong/weak router from preference data, Shnitzer et al. {shnitzer} learn selection rules from benchmarks, "
      "and Zooter {zooter} distills reward-model scores into a routing function; LLM-Blender {blender} ensembles the "
      "outputs of several models at the price of running all of them. These routers are largely supervised and fixed "
      "after training. On the bandit side, UCB strategies select the action with the highest optimistic estimate "
      "{auer}; LinUCB {lics} conditions this on context, with regret analyses by Chu et al. {chu} and improved "
      "confidence sets by Abbasi-Yadkori et al. {abbasi}. Logged data allow offline replay evaluation of bandit "
      "policies {replay}, and sliding-window and discounted UCB handle non-stationary rewards {garivier}. MixLLM "
      "{mixllm} applies contextual bandits to routing with tag-enhanced embeddings and quality and cost predictors, and "
      "PILOT {pilot} extends LinUCB with a learned query-model embedding and an online budget policy. Reward quality "
      "depends on scoring: references and unit tests are reliable where they apply {mmlu,gsm8k,humaneval}, while "
      "LLM judges show position and verbosity biases {judge}. The present work keeps the router deliberately simple and "
      "inspectable and reports measured results against simple baselines, including where it does not win."),

("H1", "III. Problem Formulation and Study Design"),
("H2", "A. Contextual-Bandit Formulation"),
("P", "Queries arrive sequentially for t = 1, ..., T. For query q_t a feature extractor produces x_t in R^d, the router "
      "chooses a model a_t from the candidate set K, and the system observes a reward r_t only for that model. The goal "
      "is to maximize cumulative reward"),
("EQ", "G(T) = Σ_{t=1..T} r_t(a_t)", 1),
("P", "or, equivalently, to minimize regret relative to the per-query oracle a*_t = argmax_a r_t(a), which is "
      "computable only because every model is run on every query offline:"),
("EQ", "Reg(T) = Σ_{t=1..T} [ r_t(a*_t) − r_t(a_t) ]", 2),
("H2", "B. Candidate Model Pool"),
("P", "The pool contains three Qwen3 models of different sizes {qwen3} (Table I). One family keeps tokenizer, prompt "
      "format, and training recipe fixed, so differences are driven mainly by scale. All three are served with vLLM "
      "{vllm} through one OpenAI-compatible interface on a single NVIDIA RTX 4090 (24 GB). Because bf16 Qwen3-14B does "
      "not fit in 24 GB, Qwen's official FP8 checkpoints are used for all three models so precision is identical across "
      "the pool. Decoding is greedy (temperature 0, seed 0) with at most 1,024 output tokens, a 120 s timeout, and "
      "thinking mode disabled {qwen3}. The models are served one at a time. Cost is a compute proxy: the per-1k-token "
      "rate in Table I times the request's input plus output tokens."),
("TABLE", "I", "CANDIDATE MODEL POOL AS RUN IN THE PILOT",
    ["Role", "Checkpoint", "Precision", "Serving / Hardware", "Cost Rate (per 1k tokens)", "Decoding"],
    [["Small", "Qwen/Qwen3-4B-FP8", "FP8", "vLLM, 1× RTX 4090", "4.0", "T = 0, ≤1,024 tokens, no thinking"],
     ["Medium", "Qwen/Qwen3-8B-FP8", "FP8", "vLLM, 1× RTX 4090", "8.0", "T = 0, ≤1,024 tokens, no thinking"],
     ["Large", "Qwen/Qwen3-14B-FP8", "FP8", "vLLM, 1× RTX 4090", "14.0", "T = 0, ≤1,024 tokens, no thinking"]],
    [900, 2100, 1000, 1900, 1500, 2723]),
("H2", "C. Task Categories and Dataset"),
("P", "The full design covers seven categories with a quality signal for each: general question answering (exact "
      "match, e.g., MMLU {mmlu}), reasoning and multi-step problems (final-answer match, e.g., GSM8K {gsm8k}), coding "
      "(unit tests, e.g., HumanEval {humaneval}), and summarization, writing, information extraction, and instruction "
      "following (rubric scores from a validated judge, field-level match, or constraint checks). The pilot reported "
      "here covers the reasoning category only: 300 problems sampled with seed 0 from the official GSM8K test split "
      "{gsm8k}, divided into 90 development and 210 held-out test queries. Quality is 1 when the final numeric answer "
      "matches the reference and 0 otherwise; the scorer reproduces all 1,319 GSM8K reference answers. Every query is "
      "run on every model (900 outcomes), and each outcome logs the response, latency, input and output tokens, cost, "
      "failure status, and quality."),

("H1", "IV. Proposed Routing Framework"),
("P", "Fig. 1 shows the loop: the query is converted into features, LinUCB scores each candidate, one model generates "
      "the response, the response is evaluated, a reward is computed, and the chosen model's statistics are updated "
      "before the next query."),
("FIG", 1, "Fig. 1. Proposed adaptive LLM routing loop with contextual-bandit feedback."),
("H2", "A. Query Features"),
("P", "The context x_t has d = 12 entries computed before generation: log prompt length, a one-hot indicator over the "
      "seven task categories, a code-content indicator, a reasoning-cue indicator, a difficulty estimate, and a constant "
      "bias. A low-dimensional sentence embedding {sbert} is an optional extension. The contribution of each group is "
      "measured by ablation (E4)."),
("H2", "B. LinUCB Scoring and Update"),
("P", "Disjoint LinUCB {lics} keeps for each model a a d×d matrix A_a and a vector b_a, initialized as"),
("EQ", "A_a = λ·I_d,   b_a = 0_d", 3),
("P", "with regularization λ > 0. The reward estimate for model a is θ_a = A_a⁻¹ b_a, and its score for context x is"),
("EQ", "p_a = θ_aᵀ x + α · √( xᵀ A_a⁻¹ x )", 4),
("P", "The first term estimates the expected reward; the second is an exploration bonus that is large when the router "
      "has little experience with similar contexts for that model, scaled by α. The router selects a_t = argmax_a p_a "
      "and, after observing r_t, updates only the chosen model:"),
("EQ", "A_a ← A_a + x_t x_tᵀ,   b_a ← b_a + r_t x_t   (a = a_t)", 5),
("H2", "C. Reward Function"),
("P", "The reward combines the competing objectives as"),
("EQ", "R = α_q · Q − β_c · C̃ − γ_l · L̃", 6),
("P", "where Q in [0, 1] is response quality (the weight α_q is distinct from the exploration coefficient α), and C̃ and L̃ are cost and latency normalized with fixed ranges from the "
      "development split and clipped to [0, 1], e.g., C̃ = clip((C − C_min)/(C_max − C_min), 0, 1). The pilot uses "
      "α_q = 1.0, β_c = 0.3, and γ_l = 0.2, with ranges C in [0.93, 6.25] and L in [2,746, 8,853] ms. Failures and "
      "timeouts receive Q = 0 with their measured cost and latency, so a cheap but unusable response cannot earn a high "
      "reward. Feedback can also come from user ratings or a judge model {judge}; judge scores are validated against "
      "manual inspection before use, and observations without a validated quality signal are excluded from updates."),
("H2", "D. Operational Safeguards"),
("P", "The router service handles timeouts, unavailable models, and malformed outputs explicitly, and writes every "
      "decision (context, selected model, outcome, reward, update) to an audit log so that runs can be replayed. Any "
      "fallback policy, such as retrying with a larger model, is evaluated and documented rather than silently "
      "changing the experiment. Algorithm 1 summarizes the procedure."),
("ALGO", "Algorithm 1. LinUCB-Based Adaptive LLM Routing", [
    "1) For every candidate model a: initialize A_a = λI_d and b_a = 0_d.",
    "2) For each incoming query q_t:",
    "3)     Extract the context vector x_t = extract_features(q_t).",
    "4)     For each model a: compute θ_a = A_a⁻¹b_a and p_a = θ_aᵀx_t + α√(x_tᵀA_a⁻¹x_t).",
    "5)     Select a_t = argmax_a p_a; call model a_t; record latency, tokens, cost, and errors.",
    "6)     Evaluate the response to obtain Q; compute r_t = α_q·Q − β_c·C̃ − γ_l·L̃ (failures: Q = 0).",
    "7)     For a = a_t only: A_a ← A_a + x_t x_tᵀ and b_a ← b_a + r_t x_t; log the decision."]),

("H1", "V. Experimental Setup"),
("P", "LinUCB is compared with four policies that use the same pool, prompts, and reward (Table II). The rule-based "
      "policy maps each task category to the model with the highest mean development reward."),
("TABLE", "II", "ROUTING POLICIES COMPARED",
    ["Policy", "Decision Rule", "Learns from Feedback"],
    [["Small-only", "Always select Qwen3-4B", "No"],
     ["Large-only", "Always select Qwen3-14B", "No"],
     ["Random", "Select a model uniformly at random (seeded)", "No"],
     ["Rule-based", "Category → best model on the development split", "No"],
     ["LinUCB (proposed)", "Select argmax of Eq. (4); update with Eq. (5)", "Yes"]],
    [2400, 5523, 2200]),
("P", "The evaluation follows Fig. 2. Because every query is run on every model, each policy is replayed on the same "
      "query stream and sees only the logged outcome of the model it selects; this also makes the oracle in Eq. (2) "
      "computable {replay}. Learning policies are first warmed up on the 90 development queries (unscored) and then "
      "evaluated on the 210 held-out queries. Each policy is run with five seeds (0-4) that change the query order and "
      "the random choices, and results are reported as mean ± standard deviation across seeds {efron}. The main "
      "comparison uses α = 1.0 and λ = 1.0. Metrics are mean quality, mean and p95 latency, cost per request, failure "
      "rate, cumulative reward, regret, and the model-selection distribution. The sweeps vary one factor at a time with "
      "α = 0.25 unless stated: reward weights (E2), feature groups (E4), the exploration coefficient α (E6), and three "
      "bandit variants (linear Thompson sampling, ε-greedy ridge regression with ε = 0.1, and sliding-window LinUCB "
      "with a window of 200). Workload shift (E5) and multi-category analysis (E3) require more than one task category "
      "and are left to the full study."),
("FIG", 2, "Fig. 2. Evaluation design: full-information logging, replay-based policy comparison, and held-out reporting."),

("H1", "VI. Results and Analysis"),
("H2", "A. Per-Model Outcomes"),
("P", "Table III summarizes the 900 logged outcomes on the held-out split. No request failed or timed out. The three "
      "models are close in accuracy: Qwen3-8B is highest at 95.7%, Qwen3-14B reaches 95.2%, and Qwen3-4B 94.8%, a gap "
      "of one to two problems out of 210. Cost per request, by contrast, is 2.2 and 3.3 times higher for the 8B and "
      "14B models. Latency does not follow size: the 14B model produced the shortest answers (231 output tokens on "
      "average) and had the lowest mean latency, so on this hardware and FP8 setup latency is governed more by answer "
      "length than by parameter count. The development split is harder for the small model (86.7% versus 92.2% and "
      "94.4%), a reminder that 90 queries give noisy estimates. On the test split at least one model is correct on "
      "209 of 210 problems; Qwen3-4B is wrong but a larger model is right on 10 of them."),
("TABLE", "III", "PER-MODEL OUTCOMES ON GSM8K (DEV n = 90, TEST n = 210) AND LINUCB SELECTION SHARE ON TEST",
    ["Model", "Accuracy Dev / Test", "Mean Latency (ms, test)", "Mean Output Tokens (test)", "Cost / Request (test)", "LinUCB Share (test)"],
    [["Qwen3-4B", "0.867 / 0.948", "5,371", "251", "1.423", "0.982"],
     ["Qwen3-8B", "0.922 / 0.957", "5,831", "280", "3.075", "0.012"],
     ["Qwen3-14B", "0.944 / 0.952", "5,235", "231", "4.706", "0.006"]],
    [1500, 1700, 1700, 1700, 1700, 1823]),
("H2", "B. Overall Comparison (E1)"),
("TABLE", "IV", "OVERALL POLICY COMPARISON ON THE HELD-OUT SPLIT (210 QUERIES, MEAN ± SD OVER 5 SEEDS)",
    ["Policy", "Mean Quality", "Mean Latency (ms)", "p95 Latency (ms)", "Cost / Request", "Failure Rate", "Cumulative Reward", "Regret"],
    [["Small-only", "0.948 ± 0.000", "5371 ± 0", "9290 ± 0", "1.423 ± 0.000", "0.000", "176.0 ± 0.0", "8.9 ± 0.0"],
     ["Large-only", "0.952 ± 0.000", "5235 ± 0", "8297 ± 0", "4.706 ± 0.000", "0.000", "139.9 ± 0.0", "45.0 ± 0.0"],
     ["Random", "0.946 ± 0.009", "5464 ± 61", "8522 ± 349", "3.131 ± 0.102", "0.000", "155.2 ± 2.9", "29.7 ± 2.9"],
     ["Rule-based", "0.948 ± 0.000", "5371 ± 0", "9290 ± 0", "1.423 ± 0.000", "0.000", "176.0 ± 0.0", "8.9 ± 0.0"],
     ["LinUCB (proposed)", "0.948 ± 0.000", "5359 ± 28", "9273 ± 39", "1.460 ± 0.009", "0.000", "175.6 ± 0.1", "9.3 ± 0.1"]],
    [1723, 1200, 1200, 1200, 1200, 1200, 1200, 1200]),
("P", "Table IV reports the main comparison. Under the default weights the cost and latency penalties outweigh the small "
      "accuracy gains of the larger models, so the 4B model is the best fixed choice. LinUCB learns this: it routes "
      "98.2% of test queries to Qwen3-4B (Table III), reaches the same mean quality (0.948), and costs 1.460 per request, "
      "69% less than large-only routing (4.706) at 0.4 percentage points lower accuracy. Its cumulative reward of "
      "175.6 ± 0.1 is 35.7 higher than large-only and 20.4 higher than random routing. It does not beat small-only or "
      "rule-based routing (176.0); its residual exploration of the larger models (1.8% of queries) costs 0.4 reward "
      "units, i.e., 0.002 per query. The rule-based policy coincides with small-only because the pilot has a single "
      "category whose best development model is Qwen3-4B. Regret against the per-query oracle stays at 8.9 even for the "
      "best fixed policy because, with binary quality, the oracle also knows which model happened to be right on each "
      "problem. RQ2 is therefore answered negatively for this workload: LinUCB matches but does not exceed the best "
      "fixed policy, which is expected when one model dominates and no context separates the queries."),
("H2", "C. Reward Weights, Exploration, Features, and Variants"),
("TABLE", "V", "SWEEPS ON THE HELD-OUT SPLIT (LINUCB UNLESS STATED; MEAN OVER 5 SEEDS)",
    ["Setting", "Mean Reward", "Mean Quality", "Cost / Request", "Regret", "Share 4B / 8B / 14B"],
    [["E2: β_c = 0, γ_l = 0", "0.948", "0.948", "1.423", "10.0", "1.00 / 0.00 / 0.00"],
     ["E2: β_c = 0.1, γ_l = 0.05", "0.918", "0.948", "1.423", "9.6", "1.00 / 0.00 / 0.00"],
     ["E2: β_c = 0.3, γ_l = 0.2", "0.838", "0.948", "1.423", "8.9", "1.00 / 0.00 / 0.00"],
     ["E2: β_c = 0.6, γ_l = 0.4", "0.687", "0.950", "1.759", "16.4", "0.80 / 0.20 / 0.00"],
     ["E2: β_c = 1.0, γ_l = 0.6", "0.610", "0.948", "1.423", "6.2", "1.00 / 0.00 / 0.00"],
     ["E6: α = 0, 0.1, 0.25", "0.838", "0.948", "1.423", "8.9", "1.00 / 0.00 / 0.00"],
     ["E6: α = 0.5", "0.799", "0.951", "2.098", "17.2", "0.60 / 0.39 / 0.01"],
     ["E6: α = 1.0", "0.836", "0.948", "1.460", "9.3", "0.98 / 0.01 / 0.01"],
     ["E6: α = 2.0", "0.813", "0.953", "1.945", "14.2", "0.75 / 0.17 / 0.07"],
     ["Linear Thompson sampling", "0.823", "0.951", "1.766", "12.0", "0.88 / 0.03 / 0.09"],
     ["ε-greedy ridge (ε = 0.1)", "0.810", "0.949", "1.887", "14.8", "0.76 / 0.21 / 0.03"],
     ["Sliding-window LinUCB", "0.838", "0.948", "1.423", "8.9", "1.00 / 0.00 / 0.00"]],
    [2900, 1300, 1300, 1300, 1100, 2223]),
("P", "Table V reports the sweeps. Reward weights (E2, RQ3): with no cost or latency penalty the best fixed policy is "
      "large-only (mean reward 0.952), yet LinUCB still selects only Qwen3-4B and reaches 0.948: initial score ties "
      "are broken toward the first model, and once Qwen3-4B has accumulated positive reward during warm-up the bonus "
      "at α = 0.25 is too small to try the larger models. For every non-zero weight setting small-only is "
      "the best fixed policy, and LinUCB matches it except at β_c = 0.6, γ_l = 0.4, where it shifted 20% of queries to "
      "Qwen3-8B and lost 0.042 per query. Exploration (E6): α ≤ 0.25 is equivalent to always choosing the 4B model; "
      "larger α buys at most 0.5 percentage points of accuracy (0.953 at α = 2.0) at 37% higher cost and lower reward, "
      "and the effect is not monotone (α = 0.5 is worse than α = 1.0), reflecting seed-to-seed variation on a small "
      "test set. Feature ablation (E4, RQ1): removing length, category, cue, or difficulty features, or keeping only "
      "the bias term, leaves every metric unchanged (mean reward 0.838), so on a single-category workload the features "
      "carry no routing signal. Bandit variants: sliding-window LinUCB equals LinUCB on this stationary stream, while "
      "linear Thompson sampling and ε-greedy explore more and lose 0.015 and 0.028 mean reward."),

("H1", "VII. Discussion and Limitations"),
("P", "The pilot supports two conclusions. First, the reward definition, not the algorithm, decides the outcome: when "
      "larger models add one or two correct answers per 210 problems at two to three times the cost, any reasonable "
      "cost weight makes the small model the right choice, and a learned router can at best match it. Second, LinUCB "
      "found that choice from 90 warm-up queries without hand-written rules, and its cost of exploration was small "
      "(0.002 reward per query). The situations in which per-query routing should pay off, namely several task "
      "categories with different best models and features that predict when the small model fails, are absent from "
      "this pilot by construction."),
("P", "The limitations follow directly. The pilot covers one dataset, one task category, and 210 test queries, so "
      "accuracy differences of under one point are within noise and the per-category analysis (E3), workload shift "
      "(E5), and the effect of the category features cannot yet be assessed. Cost is a token-based proxy rather than "
      "measured GPU time, and latency was measured with one model served at a time on one GPU, without concurrent "
      "load. Results were obtained with FP8 checkpoints and thinking mode disabled and may not transfer to bf16, "
      "reasoning mode, or other model families. The reward weights were fixed for the pilot and not tuned. Standard "
      "LinUCB assumes stationary rewards, so drifting workloads may need sliding-window or discounted variants "
      "{garivier}, and judge-based rewards for open-ended categories must be validated before use {judge}."),

("H1", "VIII. Conclusion and Future Work"),
("P", "This paper presented an adaptive LLM routing framework that formulates per-query model selection as a "
      "contextual-bandit problem and uses LinUCB to balance quality, latency, and cost without modifying the candidate "
      "models. In a GSM8K pilot with Qwen3-4B, 8B, and 14B on one RTX 4090, LinUCB learned to route 98.2% of queries to "
      "the smallest model, matched its 94.8% accuracy, and cut cost per request by 69% relative to large-only routing, "
      "but it did not exceed the small-only and rule-based baselines, which are near-optimal on a single-category "
      "workload. Future work is to run the full seven-category evaluation with workload shift, add features that "
      "predict small-model failures, measure GPU-time cost under concurrent load, compare budget-constrained routing "
      "{pilot}, and evaluate non-stationary and neural bandit variants."),
]

REFS = {
 "frugal": 'L. Chen, M. Zaharia, and J. Zou, "FrugalGPT: How to use large language models while reducing cost and improving performance," arXiv:2305.05176, 2023.',
 "hybrid": 'D. Ding et al., "Hybrid LLM: Cost-efficient and quality-aware query routing," in Proc. Int. Conf. Learning Representations (ICLR), 2024.',
 "routellm": 'I. Ong et al., "RouteLLM: Learning to route LLMs from preference data," in Proc. Int. Conf. Learning Representations (ICLR), 2025.',
 "shnitzer": 'T. Shnitzer et al., "Large language model routing with benchmark datasets," arXiv:2309.15789, 2023.',
 "zooter": 'K. Lu et al., "Routing to the expert: Efficient reward-guided ensemble of large language models," in Proc. Conf. North American Chapter of the Association for Computational Linguistics (NAACL), 2024.',
 "routerbench": 'Q. J. Hu et al., "RouterBench: A benchmark for multi-LLM routing system," arXiv:2403.12031, 2024.',
 "lattimore": 'T. Lattimore and C. Szepesvári, Bandit Algorithms. Cambridge, U.K.: Cambridge Univ. Press, 2020.',
 "lics": 'L. Li, W. Chu, J. Langford, and R. E. Schapire, "A contextual-bandit approach to personalized news article recommendation," in Proc. 19th Int. Conf. World Wide Web (WWW), 2010, pp. 661-670, doi: 10.1145/1772690.1772758.',
 "chu": 'W. Chu, L. Li, L. Reyzin, and R. E. Schapire, "Contextual bandits with linear payoff functions," in Proc. 14th Int. Conf. Artificial Intelligence and Statistics (AISTATS), PMLR vol. 15, 2011, pp. 208-214.',
 "mixllm": 'X. Wang et al., "MixLLM: Dynamic routing in mixed large language models," in Proc. 2025 Conf. Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL), vol. 1, 2025, pp. 10912-10922, doi: 10.18653/v1/2025.naacl-long.545.',
 "pilot": 'P. Panda, R. Magazine, C. Devaguptapu, S. Takemori, and V. Sharma, "Adaptive LLM routing under budget constraints," in Findings of the Association for Computational Linguistics: EMNLP 2025, 2025, pp. 23934-23949.',
 "blender": 'D. Jiang, X. Ren, and B. Y. Lin, "LLM-Blender: Ensembling large language models with pairwise ranking and generative fusion," in Proc. 61st Annu. Meeting of the Association for Computational Linguistics (ACL), 2023.',
 "auer": 'P. Auer, N. Cesa-Bianchi, and P. Fischer, "Finite-time analysis of the multiarmed bandit problem," Machine Learning, vol. 47, no. 2-3, pp. 235-256, 2002.',
 "abbasi": 'Y. Abbasi-Yadkori, D. Pál, and C. Szepesvári, "Improved algorithms for linear stochastic bandits," in Advances in Neural Information Processing Systems (NeurIPS), vol. 24, 2011.',
 "replay": 'L. Li, W. Chu, J. Langford, and X. Wang, "Unbiased offline evaluation of contextual-bandit-based news article recommendation algorithms," in Proc. 4th ACM Int. Conf. Web Search and Data Mining (WSDM), 2011.',
 "garivier": 'A. Garivier and E. Moulines, "On upper-confidence bound policies for switching bandit problems," in Proc. 22nd Int. Conf. Algorithmic Learning Theory (ALT), LNCS vol. 6925, 2011, pp. 174-188.',
 "mmlu": 'D. Hendrycks et al., "Measuring massive multitask language understanding," in Proc. Int. Conf. Learning Representations (ICLR), 2021.',
 "gsm8k": 'K. Cobbe et al., "Training verifiers to solve math word problems," arXiv:2110.14168, 2021.',
 "humaneval": 'M. Chen et al., "Evaluating large language models trained on code," arXiv:2107.03374, 2021.',
 "judge": 'L. Zheng et al., "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena," in Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, vol. 36, 2023.',
 "qwen3": 'A. Yang et al., "Qwen3 technical report," arXiv:2505.09388, 2025.',
 "vllm": 'W. Kwon et al., "Efficient memory management for large language model serving with PagedAttention," in Proc. 29th ACM Symp. Operating Systems Principles (SOSP), 2023, doi: 10.1145/3600006.3613165.',
 "sbert": 'N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in Proc. Conf. Empirical Methods in Natural Language Processing and Int. Joint Conf. Natural Language Processing (EMNLP-IJCNLP), 2019, pp. 3982-3992.',
 "efron": 'B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York, NY, USA: Chapman & Hall, 1993.',
}
