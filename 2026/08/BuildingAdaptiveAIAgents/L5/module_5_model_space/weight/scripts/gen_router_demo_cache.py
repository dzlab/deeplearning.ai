"""(Re)generate router_demo_cache.json for L5_lab.show_adapter_router_demo.

The router demo (base-vs-superpolite plug-n-play routing) replays base + v2
generations for a fixed set of queries from this committed cache, so the notebook
cell is reproducible with no GPU. Run this to (re)create it — cache-first, so it
only generates missing keys:

    # GPU box with the polite adapters, OR the 4-bit GGUF CPU path:
    python module_5_model_space/weight/scripts/gen_router_demo_cache.py

Writes router_demo_cache.json next to the polite cache; commit it to BOTH repos
(dl-ai module_5_model_space/weight/data/ and SC-Oracle-C2 L5/ro_shared_data/).
Keep DEMO byte-identical to L5_lab.show_adapter_router_demo's DEMO list.
"""
from course_lab import paths
from course_lab.gemma_inference import generate_gemma

DEMO = [
    "My unit test keeps failing and I'm completely stuck, what should I check?",
    "I've been debugging this regex for an hour and I'm losing my mind, can you help?",
    "Can you help me name this variable?",
    "What is the time complexity of binary search?",
    "List the idempotent HTTP methods.",
    "Explain how a hash map works.",
]


def main() -> None:
    adapters = {l: paths.polite_adapter_dir(l) for l in ("v1", "v2")}
    cache_path = paths.polite_cached_responses_json().parent / "router_demo_cache.json"
    # generate_gemma is cache-first: reuses cache_path, generates only misses,
    # and writes results back under {arm}::sha256(prompt) keys.
    generate_gemma("base", DEMO, adapters=adapters, cache_path=cache_path, max_new_tokens=80)
    generate_gemma("v2", DEMO, adapters=adapters, cache_path=cache_path, max_new_tokens=80)
    print(f"wrote {cache_path}")


if __name__ == "__main__":
    main()
