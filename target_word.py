from gensim.models import Word2Vec
import heapq
from tqdm import tqdm

# Word2Vecモデルのラウトを設定
model_path = 'models/sw_200d_2w/ja-gensim.200d.2w.data.sw.model'

print("Loading Word2Vec model...")  # Inform user about loading process
model = Word2Vec.load(model_path)
print("Model loaded successfully!\n")


# 単語類似度の数値(similarity_targets)と各々の数値に対する単語の回答数(top_n)を設定
def find_closest_words_full_vocab(target_word, similarity_targets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], top_n=3):
    try:
        print(f"Processing similarity scores for '{target_word}'...\n")

        # Get all words in the vocabulary
        vocab = model.wv.index_to_key
        total_words = len(vocab)

        # Compute similarity scores for all words in the vocabulary with a progress bar
        similarities = []
        for word in tqdm(vocab, desc="Computing similarities", unit="word"):
            if word != target_word:  # Exclude the target word itself
                try:
                    score = model.wv.similarity(target_word, word)
                    similarities.append((word, score))
                except KeyError:
                    continue  # Ignore words not found in vocabulary

        print("\nSorting similarity scores...")
        similarities.sort(key=lambda x: x[1])  # Sort by similarity score (ascending)

        # Find the closest words for each target similarity
        closest_words = {}
        for sim_target in similarity_targets:
            # Use a min-heap to find the closest `top_n` words to the target similarity
            closest = heapq.nsmallest(top_n, similarities, key=lambda x: abs(x[1] - sim_target))
            closest_words[sim_target] = closest

        print("Processing completed!\n")
        return closest_words

    except KeyError:
        print(f"'{target_word}' is not in the model's vocabulary.")
        return None


# ターゲットの単語を入力
target_word = "りんご"
results = find_closest_words_full_vocab(target_word)

if results:
    for sim_target, words in results.items():
        print(f"Similarity {sim_target}:")
        for word, score in words:
            print(f"  Word = {word}, Exact Score = {score}")
