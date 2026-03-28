from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch

def load_data():
    # Example: small NLP dataset
    from sklearn.datasets import fetch_20newsgroups
    data = fetch_20newsgroups(subset='all', categories=['sci.space', 'rec.autos'])
    X_text, y = data.data, data.target

    # Convert raw text to embeddings
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(X_text).toarray()  # shape: (n_samples, 500)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    import torch

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test