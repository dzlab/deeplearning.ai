# Building Adaptive AI Agents — Course Glossary

Key terms for building AI agents that improve over time through behavior adaptation, knowledge adaptation, and weight adaptation.

---

## Foundations

**Agent** A system that takes in information from its environment, reasons about what to do next using an LLM, and uses tools to act on that environment. An agent that also has memory can retain useful experience from past interactions instead of starting fresh each time.

**Harness** The software scaffolding that surrounds an LLM to turn it into a working agent. It assembles the context the model sees, gives the model the ability to call tools, records results to memory, and re-invokes the model within the agent loop. The LLM supplies the reasoning; the harness supplies the structure around it.

**Agent Loop** The repeating cycle by which an agent completes a task: a user request triggers the model to decide on an action, the harness executes a tool and returns the result, and that result is added to the context for the next iteration. The loop continues until the task succeeds or the agent stops.

**Trace** A complete record of everything that happened during an agent's run on a task — its conversations, tool calls, decisions, errors, and fixes. Traces are the raw material used to derive skills and other forms of long-term memory.

**Working Memory** The messages and tool results currently in use within a single agent run. It exists only for the duration of that run and does not persist afterward.

**Episodic Memory** The record of what happened in previous agent runs. Unlike working memory, episodic memory persists across sessions, allowing an agent to recall specific past interactions rather than just the current one.

**Semantic Memory** Project-specific facts and knowledge an agent holds about its environment (for example, the structure or conventions of a codebase), as distinct from episodic memory's record of events, semantic memory stores facts.

**Procedural Memory** The workflow steps or reusable methods an agent has learned for accomplishing a task. Where semantic memory captures what is true, procedural memory captures how to do something, this is the form of memory that skills (see Skill Induction) are built from.

**Token Space Adaptation** Improving an agent's performance by changing the context and instructions it receives, without altering the underlying model's weights. It covers two methods: behavior adaptation (skill induction) and knowledge adaptation (code knowledge graphs). Contrast with Weight Space Adaptation, which modifies the model itself and is generally used only once token space methods are exhausted.

**Weight Space Adaptation** Improving an agent's behavior by directly modifying the weights of the underlying model, typically through fine-tuning. It is more costly than token space adaptation and is used to patch specific behaviors (such as tone or refusals) that context alone cannot reliably produce.

## Behavior Adaptation: Skill Induction

**Skill Induction** The process of turning an agent's repeated experiences, captured in its traces, into reusable procedures called skills, so future runs can retrieve a proven approach instead of rediscovering it from scratch.

**Skill Induction Engine** The component (built on an LLM) that reviews an agent's traces, identifies repeated failures and fixes, and drafts or proposes an enhanced version of a skill along with supporting evidence for the change.

**Skill** A short, reusable procedure, typically written as a markdown file, that describes steps, tools, common errors, and fixes for accomplishing a recurring task. Skills are produced by the Skill Induction Engine and, once approved, are retrieved by the agent whenever it faces a matching task.

**Skill Box** The managed, versioned collection of approved skills available for an agent to retrieve and use. A skill only enters the Skill Box after human review; each version promoted into it is considered active until replaced.

**Human-in-the-Loop Review** The approval step where a person reviews a proposed skill enhancement and either approves it (promoting it to active status in the Skill Box) or rejects it with a reason. This gate exists because approval turns a proposal into behavior the agent will repeat on every matching task going forward, approving a flawed or maliciously planted procedure would let a "poisoned" trace become standing agent behavior, and every approval or rejection also creates an accountable owner for that skill.

## Knowledge Adaptation: Code Knowledge Graph

**Retrieval Bottleneck** The observation that in large, unstructured code bases, the hardest and most time-consuming part of a coding agent's task is usually not writing or generating code, but finding the correct files to read or modify in the first place.

**Keyword/Regex Search** A traditional code-search method based on matching literal keywords or regular expression patterns in file contents. It is fast but structurally blind: it can find files containing a matching term while missing related files that don't share that exact wording, the gap that the Code Knowledge Graph is designed to close.

**Code Knowledge Graph** A property graph representation of a code base that stores relationships between its files and functions, specifically import, call, and co-edit relationships, so an agent can retrieve related code by traversing structure rather than by keyword matching alone.

**Import Relationship (Import Edge)** A file-to-file connection in the Code Knowledge Graph indicating that one file imports from another. It is the most basic relationship type and outlines which files may be affected when a dependency changes.

**Call Relationship (Call Edge)** A function-to-function connection in the Code Knowledge Graph indicating that one function calls another, often across files. It flags which functions may need review when a called function's behavior changes.

**Co-edit Relationship (Co-edit Edge)** A file-to-file connection in the Code Knowledge Graph derived from a repository's Git history, linking files that have tended to be modified together in the same commits, even when they share no direct import or call relationship.

**Property Graph** The underlying graph database structure used to store the Code Knowledge Graph, made up of vertices (files and functions, each with a type and label) and directed edges (import, call, and co-edit relationships) connecting them.

**Deduplication (Graph Auditing)** The step of identifying and removing near-duplicate or highly similar nodes from the Code Knowledge Graph before retrieval, so that redundant files don't inflate results or create misleading duplicate edges.

**Anchor** The first node in the Code Knowledge Graph found to be semantically similar to an incoming query. It serves as the starting point from which the retrieval algorithm (PageRank) expands outward through the graph.

**PageRank Retrieval** A ranking algorithm applied to the Code Knowledge Graph that scores nodes by their importance relative to an anchor node, expanding outward along edges. Unlike plain graph traversal, which gives every node at the same distance from the anchor equal weight, PageRank produces a custom score per node based on a combination of distance and relationship structure, surfacing the most relevant files even when they are several hops from the anchor.

**Multi-hop Question** A query whose correct answer lies in a node that is more than one relationship away from the anchor node, the case where keyword search is most likely to miss the answer and graph-based retrieval shows the clearest advantage.

## Weight Space Adaptation: Fine-Tuning

**Fine-Tuning** The process of further training a pre-trained model's weights to patch a specific behavior, such as tone, formatting, or refusing certain topics — after the model has already been trained on general data. Because foundation models are trained on broad data already, fine-tuning is generally used to adjust how a model behaves rather than to add new factual knowledge.

**Adapter** A small, separately trained add-on placed on top of a frozen base model's weights to produce a targeted behavior change (for example, a more polite tone) without altering the original model. A router can select among multiple trained adapters, or the base model itself, depending on the incoming query.

**LoRA (Low-Rank Adaptation)** A parameter-efficient fine-tuning technique that freezes a model's original weights and adds two small trainable matrices per targeted layer whose product approximates the desired weight change. Only these matrices are trained and added to the frozen weights at inference time, which keeps training cheap; the technique typically targets around 1% of a model's total weights.

**Quantization** The process of reducing the numerical precision used to represent a model's weights (for example, from 32-bit floating point down to 4-bit values) to shrink memory and storage requirements. Converting back to higher precision afterward introduces some reconstruction error, trading a small loss of precision for a large reduction in size.

**Catastrophic Forgetting** A failure mode of fine-tuning in which training too many of a model's weights overwrites what the original network had already learned, degrading its broader capabilities. It is the risk on the opposite end from training too few weights, which instead fails to produce enough of the desired behavior change, fine-tuning aims for the smallest weight change that reliably produces the target behavior.

**Router (Model Router)** A component that inspects an incoming query, for example using pattern or regex matching, and decides which model should answer it: the base model, or one of several fine-tuned adapters. It lets a system serve a factual, neutral response to one query and a stylistically adapted response to another, without a human choosing the model manually.

## Applied Examples

**Skeleton Import Graph** The set of import relationships within a Code Knowledge Graph, described as "the skeleton" because it determines the base paths an agent walks to find everything a given file depends on, before call and co-edit relationships add further connections.

**Win Rate (Task Efficiency Comparison)** The proportion of coding tasks, out of a benchmark set, on which an agent performs better (faster, fewer steps, or fewer tokens) using one method, such as Code Knowledge Graph retrieval, compared to a baseline such as keyword search or no additional context.