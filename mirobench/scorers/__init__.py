"""MiroBench scoring modules.

Nine scorer families covering lexical diversity, semantic similarity, narrativity,
toxicity, emotion, politeness, disagreement, and thread structure.
"""

SCORER_SCRIPTS = [
    "score_thread_disagreement",
    "score_thread_self_bleu",
    "score_thread_self_bertscore",
    "score_thread_semantic_uniformity",
    "score_thread_storyseeker",
    "score_thread_go_emotions",
    "score_thread_politeness",
    "score_thread_structure",
    "score_thread_detoxify",
]

SCORER_TO_RESULT_FILE = {
    "score_thread_disagreement": "stance_disagreement_results.json",
    "score_thread_self_bleu": "self_bleu_results.json",
    "score_thread_self_bertscore": "self_bertscore_results.json",
    "score_thread_semantic_uniformity": "semantic_uniformity_results.json",
    "score_thread_storyseeker": "storyseeker_results.json",
    "score_thread_go_emotions": "go_emotions_results.json",
    "score_thread_politeness": "politeness_results.json",
    "score_thread_structure": "thread_structure_results.json",
    "score_thread_detoxify": "detoxify_results.json",
}
