import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import nltk
import numpy as np
import requests
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


WIKI_URL = "https://en.wikipedia.org/wiki/Gundam"
SEED = 42
VECTOR_SIZE = 100
MIN_COUNT = 1
EPOCHS = 150


def ensure_nltk() -> None:
    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource)


def fetch_wikipedia_article(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; Unit5-SGNS/1.0)",
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    content = soup.find("div", {"id": "mw-content-text"})
    if content is None:
        raise ValueError("Wikipedia article content not found.")

    text_blocks = []
    for p in content.find_all(["p", "li"]):
        text = p.get_text(" ", strip=True)
        if text:
            text_blocks.append(text)

    text = "\n".join(text_blocks)
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str) -> List[List[str]]:
    sentences = sent_tokenize(text)
    processed: List[List[str]] = []

    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub(r"[^a-z0-9\-\s]", " ", sentence)
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if not sentence:
            continue

        tokens = word_tokenize(sentence)
        cleaned: List[str] = []
        for token in tokens:
            token = token.strip("-")
            if not token or token.isdigit() or len(token) < 2:
                continue
            cleaned.append(token)

        if len(cleaned) >= 3:
            processed.append(cleaned)

    return processed


def corpus_stats(sentences: List[List[str]]) -> Dict[str, int]:
    flat = [token for sentence in sentences for token in sentence]
    return {
        "num_sentences": len(sentences),
        "num_tokens": len(flat),
        "vocab_size": len(set(flat)),
    }


def train_sgns(sentences: List[List[str]], window: int) -> Word2Vec:
    return Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=window,
        min_count=MIN_COUNT,
        workers=4,
        sg=1,
        negative=10,
        epochs=EPOCHS,
        sample=1e-3,
        alpha=0.025,
        min_alpha=0.0007,
        seed=SEED,
    )


def has_word(model: Word2Vec, word: str) -> bool:
    return word in model.wv.key_to_index


def cosine(model: Word2Vec, w1: str, w2: str) -> float:
    v1 = model.wv[w1].reshape(1, -1)
    v2 = model.wv[w2].reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0][0])


def evaluate_relatedness(model: Word2Vec, test_pairs: List[Tuple[str, str, float]]) -> Dict:
    covered = []
    for w1, w2, gold_score in test_pairs:
        if has_word(model, w1) and has_word(model, w2):
            pred_score = cosine(model, w1, w2)
            covered.append((w1, w2, gold_score, pred_score))
    return {
        "covered_items": covered,
        "coverage": len(covered),
        "total": len(test_pairs),
    }


def evaluate_analogies(model: Word2Vec, analogies: List[Tuple[str, str, str, str]]) -> Dict:
    covered = 0
    correct = 0
    details = []

    for a, b, c, d in analogies:
        if not all(has_word(model, w) for w in [a, b, c, d]):
            continue

        covered += 1
        preds = model.wv.most_similar(positive=[b, c], negative=[a], topn=5)
        predicted_words = [w for w, _ in preds]
        hit = d in predicted_words
        correct += int(hit)
        details.append(
            {
                "analogy": f"{a}:{b}::{c}:?",
                "expected": d,
                "predictions": predicted_words,
                "correct_in_top5": hit,
            }
        )

    return {
        "coverage": covered,
        "total": len(analogies),
        "accuracy_top5": (correct / covered) if covered else float("nan"),
        "details": details,
    }


def collect_neighbors(model: Word2Vec, words: List[str], topn: int = 8) -> Dict[str, List[Tuple[str, float]]]:
    output: Dict[str, List[Tuple[str, float]]] = {}
    for word in words:
        if has_word(model, word):
            output[word] = model.wv.most_similar(word, topn=topn)
        else:
            output[word] = []
    return output


def print_neighbors(neighbors: Dict[str, List[Tuple[str, float]]]) -> None:
    print("\n=== Nearest Neighbors ===")
    for word, entries in neighbors.items():
        print(f"\n{word}:")
        if not entries:
            print("  [OOV]")
            continue
        for neighbor, score in entries:
            print(f"  {neighbor:20s} {score:.4f}")


def print_relatedness(rel: Dict) -> None:
    print("\n=== Relatedness Test Set ===")
    print(f"Coverage: {rel['coverage']}/{rel['total']}")
    for w1, w2, gold, pred in rel["covered_items"]:
        print(f"{w1:10s} - {w2:10s} | gold={gold:.2f} pred={pred:.4f}")


def print_analogies(analogy: Dict) -> None:
    print("\n=== Analogy Test Set ===")
    print(f"Coverage: {analogy['coverage']}/{analogy['total']}")
    print(f"Top-5 accuracy: {analogy['accuracy_top5']}")
    for item in analogy["details"]:
        print(json.dumps(item, ensure_ascii=False))


def print_direct_similarity(direct: List[Tuple[str, str, float]]) -> None:
    print("\n=== Direct Similarity Checks ===")
    for w1, w2, score in direct:
        if score is None:
            print(f"{w1:10s} <-> {w2:10s}: OOV")
        else:
            print(f"{w1:10s} <-> {w2:10s}: {score:.4f}")


def evaluate_model(model: Word2Vec) -> Dict:
    probe_words = [
        "gundam",
        "mobile",
        "suit",
        "anime",
        "series",
        "film",
        "gunpla",
        "franchise",
        "robot",
        "war",
    ]
    relatedness_test = [
        ("gundam", "gunpla", 0.95),
        ("gundam", "mobile", 0.80),
        ("gundam", "suit", 0.85),
        ("anime", "series", 0.90),
        ("film", "movie", 0.90),
        ("robot", "mecha", 0.95),
        ("franchise", "series", 0.80),
        ("war", "battle", 0.75),
        ("gundam", "kitchen", 0.05),
        ("anime", "tractor", 0.02),
        ("gunpla", "movie", 0.25),
        ("robot", "pilot", 0.45),
    ]
    analogy_test = [
        ("movie", "film", "tv", "series"),
        ("robot", "mecha", "war", "battle"),
        ("anime", "series", "movie", "film"),
        ("pilot", "cockpit", "driver", "car"),
    ]
    direct_pairs = [
        ("gundam", "gunpla"),
        ("gundam", "anime"),
        ("robot", "mecha"),
        ("gundam", "kitchen"),
    ]

    direct = []
    for w1, w2 in direct_pairs:
        if has_word(model, w1) and has_word(model, w2):
            direct.append((w1, w2, cosine(model, w1, w2)))
        else:
            direct.append((w1, w2, None))

    custom_eval_words = [
        "gundam",
        "mobile",
        "suit",
        "anime",
        "series",
        "film",
        "movie",
        "robot",
        "mecha",
        "pilot",
    ]

    return {
        "probe_words": probe_words,
        "neighbors": collect_neighbors(model, probe_words, topn=8),
        "relatedness": evaluate_relatedness(model, relatedness_test),
        "analogy": evaluate_analogies(model, analogy_test),
        "direct_similarity": direct,
        "custom_eval_words": custom_eval_words,
        "custom_eval_neighbors": collect_neighbors(model, custom_eval_words, topn=5),
    }


def pca_plot(model: Word2Vec, words: List[str], output_path: Path) -> List[str]:
    in_vocab = [w for w in words if has_word(model, w)]
    vectors = np.array([model.wv[w] for w in in_vocab])
    reduced = PCA(n_components=2, random_state=SEED).fit_transform(vectors)

    plt.figure(figsize=(12, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=36)
    for i, word in enumerate(in_vocab):
        plt.annotate(word, (reduced[i, 0], reduced[i, 1]), fontsize=9)
    plt.title("PCA projection of Word2Vec vectors (window=10)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    return in_vocab


def summarize_comparison(old_eval: Dict, new_eval: Dict) -> Dict:
    old_cov = old_eval["relatedness"]["coverage"]
    new_cov = new_eval["relatedness"]["coverage"]

    old_analogy_acc = old_eval["analogy"]["accuracy_top5"]
    new_analogy_acc = new_eval["analogy"]["accuracy_top5"]

    return {
        "relatedness_coverage_old": old_cov,
        "relatedness_coverage_new": new_cov,
        "analogy_accuracy_old": old_analogy_acc,
        "analogy_accuracy_new": new_analogy_acc,
    }


def write_text_report(
    report_path: Path,
    wiki_url: str,
    stats: Dict[str, int],
    baseline_window: int,
    new_window: int,
    old_eval: Dict,
    new_eval: Dict,
    comparison: Dict,
    pca_words_used: List[str],
) -> None:
    lines = []
    lines.append("Exercise for Unit 5 - Completed")
    lines.append("")
    lines.append(f"Wikipedia article used: {wiki_url}")
    lines.append("")
    lines.append("1) Use a Wikipedia article as dataset")
    lines.append("- Selected corpus: Gundam Wikipedia article")
    lines.append(
        "- Code part: WIKI_URL variable and article download in fetch_wikipedia_article, plus main call sequence"
    )
    lines.append("")
    lines.append("2) Preprocess the text")
    lines.append("- Preprocessing was performed by preprocess_text")
    lines.append("- Corpus stats after preprocessing:")
    lines.append(f"  num_sentences: {stats['num_sentences']}")
    lines.append(f"  num_tokens: {stats['num_tokens']}")
    lines.append(f"  vocab_size: {stats['vocab_size']}")
    lines.append("")
    lines.append("3) Train Skip-gram with Negative Sampling")
    lines.append(
        f"- Baseline model: vector_size={VECTOR_SIZE}, window={baseline_window}, sg=1, negative=10, epochs={EPOCHS}"
    )
    lines.append("- Evaluation words were changed for this run to:")
    lines.append(f"  {new_eval['custom_eval_words']}")
    lines.append("")
    lines.append("4) Report neighbors, similarity scores, and test-set performance")
    lines.append("Window=5 (OLD) summary")
    lines.append(
        f"- Relatedness coverage: {old_eval['relatedness']['coverage']}/{old_eval['relatedness']['total']}"
    )
    lines.append(f"- Analogy top-5 accuracy: {old_eval['analogy']['accuracy_top5']}")
    lines.append("- Direct similarity checks:")
    for w1, w2, score in old_eval["direct_similarity"]:
        if score is None:
            lines.append(f"  {w1} <-> {w2}: OOV")
        else:
            lines.append(f"  {w1} <-> {w2}: {score:.4f}")

    lines.append("")
    lines.append("Window=10 (NEW) summary")
    lines.append(
        f"- Relatedness coverage: {new_eval['relatedness']['coverage']}/{new_eval['relatedness']['total']}"
    )
    lines.append(f"- Analogy top-5 accuracy: {new_eval['analogy']['accuracy_top5']}")
    lines.append("- Direct similarity checks:")
    for w1, w2, score in new_eval["direct_similarity"]:
        if score is None:
            lines.append(f"  {w1} <-> {w2}: OOV")
        else:
            lines.append(f"  {w1} <-> {w2}: {score:.4f}")

    lines.append("")
    lines.append("5) Comparison (OLD vs NEW) and derived insight")
    lines.append(
        f"- Relatedness coverage: {comparison['relatedness_coverage_old']} -> {comparison['relatedness_coverage_new']}"
    )
    lines.append(
        f"- Analogy top-5 accuracy: {comparison['analogy_accuracy_old']} -> {comparison['analogy_accuracy_new']}"
    )
    lines.append(
        "- Derivation: a larger window uses wider context, which can improve topical/semantic neighborhood quality, but may blur tight syntactic relations depending on corpus size."
    )
    lines.append("")
    lines.append("6) PCA visualization")
    lines.append("- Graph file: unit5_pca_window10.png")
    lines.append(f"- Number of plotted known words: {len(pca_words_used)}")
    lines.append(f"- Words used: {pca_words_used}")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    ensure_nltk()

    raw_text = fetch_wikipedia_article(WIKI_URL)
    sentences = preprocess_text(raw_text)
    stats = corpus_stats(sentences)

    print("=== Corpus Stats ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    baseline_window = 5
    new_window = 10

    model_old = train_sgns(sentences, window=baseline_window)
    model_new = train_sgns(sentences, window=new_window)

    old_eval = evaluate_model(model_old)
    new_eval = evaluate_model(model_new)
    comparison = summarize_comparison(old_eval, new_eval)

    print("\n--- OLD MODEL (window=5) ---")
    print_neighbors(old_eval["neighbors"])
    print_relatedness(old_eval["relatedness"])
    print_analogies(old_eval["analogy"])
    print_direct_similarity(old_eval["direct_similarity"])

    print("\n--- NEW MODEL (window=10) ---")
    print_neighbors(new_eval["neighbors"])
    print_relatedness(new_eval["relatedness"])
    print_analogies(new_eval["analogy"])
    print_direct_similarity(new_eval["direct_similarity"])

    words_for_pca = [
        "gundam",
        "mobile",
        "suit",
        "anime",
        "series",
        "film",
        "movie",
        "robot",
        "mecha",
        "pilot",
        "franchise",
        "war",
        "battle",
        "television",
        "manga",
        "model",
        "plastic",
        "military",
        "space",
        "japan",
        "characters",
        "story",
        "universe",
        "fans",
        "production",
    ]
    pca_words_used = pca_plot(model_new, words_for_pca, Path("unit5_pca_window10.png"))

    model_old.save("exercise_5_skipgram_sgns_window5.model")
    model_new.save("exercise_5_skipgram_sgns_window10.model")

    result = {
        "wiki_url": WIKI_URL,
        "stats": stats,
        "window5": old_eval,
        "window10": new_eval,
        "comparison": comparison,
        "pca_words_used": pca_words_used,
    }
    Path("unit5_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved model to: exercise_5_skipgram_sgns_window5.model")
    print("Saved model to: exercise_5_skipgram_sgns_window10.model")
    print("Saved results to: unit5_results.json")

    write_text_report(
        report_path=Path("Unit 5 Exercise"),
        wiki_url=WIKI_URL,
        stats=stats,
        baseline_window=baseline_window,
        new_window=new_window,
        old_eval=old_eval,
        new_eval=new_eval,
        comparison=comparison,
        pca_words_used=pca_words_used,
    )

    print("Saved report to: Unit 5 Exercise")
    print("Saved PCA graph to: unit5_pca_window10.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
