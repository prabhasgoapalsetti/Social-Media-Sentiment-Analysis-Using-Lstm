# train.py
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle

# ================= CONFIG =================
DATA_PATH = r"C:\Users\MUKESH\Downloads\archive (1)\sentimentdataset.csv"
MAX_WORDS = 10000
MAX_LEN = 100
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 30
VALIDATION_SPLIT = 0.2
MODEL_SAVE_PATH = "sentiment_lstm_model.keras"
MIN_SAMPLES_PER_CLASS = 5
# ==========================================

# NLTK
nltk.download('punkt')
nltk.download('stopwords')

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

# Drop useless columns
cols_to_drop = ['Unnamed: 0', 'Index']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

df = df[['Text', 'Sentiment']].dropna()

# 2. Clean text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['Text'].apply(clean_text)

# 3. REMOVE RARE CLASSES FIRST
class_counts = df['Sentiment'].value_counts()
valid_classes = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index
df = df[df['Sentiment'].isin(valid_classes)]

# 🔥 IMPORTANT LINE (FIX)
le = LabelEncoder()
df['label'] = le.fit_transform(df['Sentiment'])

num_classes = df['label'].nunique()
print("Final number of classes:", num_classes)

# 4. Tokenization
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_text'])

sequences = tokenizer.texts_to_sequences(df['clean_text'])
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
y = df['label'].values

# 5. Train / Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=VALIDATION_SPLIT,
    random_state=42,
    stratify=y
)

# 6. Model
model = Sequential([
    Embedding(MAX_WORDS, EMBEDDING_DIM),
    SpatialDropout1D(0.2),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# 7. Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# 8. Train
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# 9. Save tokenizer & encoder
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Training completed successfully.")
