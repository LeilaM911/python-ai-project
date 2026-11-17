from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# داده های نمونه
texts = [
    "این فیلم عالی بود",
    "من این محصول را دوست دارم",
    "امروز خیلی خوشحالم",
    "این فیلم بسیار بد بود",
    "از خرید این محصول ناراضی هستم",
    "امروز حالم اصلا خوب نیست"
]

labels = [
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative"
]

# ساخت مدل یادگیری ماشین
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression())
])

# آموزش مدل
model.fit(texts, labels)

# آزمایش مدل
new_text = ["این محصول فوق العاده است"]
prediction = model.predict(new_text)

print("نتیجه تحلیل احساسات:", prediction[0])
