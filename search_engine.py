from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class ImageSearchEngine:
    def __init__(self, surrogates_file):
        self.filenames = []
        self.captions = []
        with open(surrogates_file, "r", encoding='utf-8') as f:
            for line in f:
                filename, caption = line.strip().split("\t")
                self.filenames.append(filename)
                self.captions.append(caption)

        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.captions)

    def search(self, query, top_k=5):
        query_vec = self.vectorizer.transform([query])
        scores = np.dot(self.tfidf_matrix, query_vec.T).toarray().flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        results = [(self.filenames[i], self.captions[i], scores[i]) for i in top_indices]
        return results

# Example usage:
if __name__ == "__main__":
    search_engine = ImageSearchEngine("image_surrogates.txt")
    results = search_engine.search("happy dog playing", top_k=5)
    for filename, caption, score in results:
        print(f"{filename} | {caption} | Score: {score:.4f}")
