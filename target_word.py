from gensim.models import Word2Vec
import heapq
import pandas as pd
from tqdm import tqdm

# Word2Vecモデルのパスを設定
model_path = 'models/sw_200d_2w/ja-gensim.200d.2w.data.sw.model'

print("Loading Word2Vec model...")  # Inform user about loading process
model = Word2Vec.load(model_path)
print("Model loaded successfully!\n")


def find_closest_words_full_vocab(target_word, target_scores=[0.1525, 0.2860, 0.4195], top_n=15):
    try:
        print(f"Processing similarity scores for '{target_word}'...\n")

        # Get all words in the vocabulary
        vocab = model.wv.index_to_key

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
        for target_score in target_scores:
            # Use a min-heap to find the closest `top_n` words to the target similarity
            closest = heapq.nsmallest(top_n, similarities, key=lambda x: abs(x[1] - target_score))
            closest_words[target_score] = closest

        print("Processing completed!\n")
        return closest_words

    except KeyError:
        print(f"'{target_word}' is not in the model's vocabulary.")
        return None


def save_results_to_excel(target_word, results):
    file_name = f"result_{target_word}.xlsx"
    writer = pd.ExcelWriter(file_name, engine='openpyxl')

    data = []

    for sim_target, words in results.items():
        data.append([f"ターゲットスコア：{sim_target}", "", ""])  # Add header row
        data.append(["単語1", "単語2", "単語類似度"])  # Add column labels

        for word, score in words:
            data.append([target_word, word, score])  # Add word similarity data

        data.append(["", "", ""])  # Add an empty row for spacing

    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.to_excel(writer, index=False, header=False, sheet_name="Results")

    writer.close()
    print(f"Results saved to {file_name}")


# ターゲットの単語を入力
target_word = "ビール"
results = find_closest_words_full_vocab(target_word)

if results:
    save_results_to_excel(target_word, results)
    for sim_target, words in results.items():
        print(f"Similarity {sim_target}:")
        for word, score in words:
            print(f"  Word = {word}, Exact Score = {score}")
