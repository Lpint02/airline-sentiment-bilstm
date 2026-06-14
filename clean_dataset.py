"""
Data Cleaning Pipeline – Kaggle Airline Tweet Sentiment
=============================================================
Pipeline riadattata per il dataset "Tweets.csv" da 14.6k righe.
Include Text Normalization (slang), lookahead negativo,
salvataggio delle emoticon e tokenizzazione custom.
"""

import pandas as pd
import re
import emoji

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
# Leggiamo il nuovo file CSV di Kaggle
df = pd.read_csv("Tweets.csv")
print(f"Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")

# Selezioniamo solo le colonne utili e le rinominiamo per comodità
df = df[['text', 'airline_sentiment']].copy()
df.rename(columns={'text': 'tweet_text'}, inplace=True)

# ── 2. MAPPARE IL SENTIMENT ──────────────────────────────────────────────────
# Trasformiamo le stringhe in numeri: negative=0, neutral=1, positive=2
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['tweet_sentiment_value'] = df['airline_sentiment'].map(sentiment_map)

# Ora possiamo scartare la colonna testuale originale
df.drop(columns=['airline_sentiment'], inplace=True)

# ── 3. PRE-COMPILED REGEX PATTERNS ───────────────────────────────────────────
_SLANG_PATTERNS = [
    (re.compile(r'\bb4\b'), 'before'),
    (re.compile(r'\bsome1\b'), 'someone'),
    (re.compile(r'\bno1\b'), 'no one'),
    (re.compile(r'\bpls\b'), 'please'),
    (re.compile(r'\bplz\b'), 'please'),
    (re.compile(r'\bthx\b'), 'thanks'),
    (re.compile(r'\bu\b(?!\s*\.)'), 'you'),
    (re.compile(r'\bur\b'), 'your'),
    (re.compile(r'\bhr\b'), 'hour'),
    (re.compile(r'\bhrs\b'), 'hours'),
    (re.compile(r'\bmin\b'), 'minute'),
    (re.compile(r'\bmins\b'), 'minutes')
]

# urls
_RE_URL = re.compile(r"http\S+|www\S+")

# mentions
_RE_MENTION = re.compile(r"@\w+")

# html entity: amp;
_RE_AMP = re.compile(r"amp;")

# punctuation spacing
_RE_PUNCT_SPACE = re.compile(r"(?<=[.,!?])(?=[a-zA-Z])")

# Financial / decimal numbers (protect before time extraction)
_RE_PRICE = re.compile(r"\b\d+\.\d+\b")

# Time formats: 18:00, 18.00, 10am, 9 pm, 9.10pm, 10:30 PM …
_RE_TIME = re.compile(r"\b\d{1,2}[:.]\d{2}(?:\s?[ap]m)?\b|\b\d{1,2}\s?[ap]m\b")

# airlines
_RE_AIRLINES = re.compile(
    r"airfrance\w*|\w*airfrance|delta|klm|ryanair|americanair|aireuropa|jetairways|lufthansa|virginamerica|united|southwestair|usairways"
) 

# Aircraft models (b777-300er, a380) and flight codes (af1440)
_RE_AIRCRAFT = re.compile(r"\b[a-z]{0,2}\d{3,4}(?:-\d{3,4}er|[a-z]{1,2})?\b")

# Any remaining isolated numbers
_RE_NUMBER = re.compile(r"\b\d+\b")

# Repeated punctuation: ??? → ?, !!! → !, ... → .
_RE_MULTI_PUNCT = re.compile(r"(\?+)|(!+)|(\.+)")

# Two or more consecutive whitespace characters → single space
_RE_EXTRA_SPACE = re.compile(r"\s{2,}")

# ── 4. CLEANING FUNCTION ─────────────────────────────────────────────────────
def _reduce_punct(match: re.Match) -> str:
    if match.group(1): return "?"
    if match.group(2): return "!"
    return "."

def clean_tweet(text: str) -> str:
    text = str(text).lower()

    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.replace("_", " ")

    for pat, repl in _SLANG_PATTERNS:
        text = pat.sub(repl, text)

    text = _RE_URL.sub("", text)
    text = _RE_MENTION.sub("", text)
    text = _RE_AMP.sub("and", text)
    text = _RE_PUNCT_SPACE.sub(" ", text)
    
    # Salva emoticon
    text = re.sub(r':\)|:-\)|;\)', ' goodemoticon ', text)
    text = re.sub(r':\(|:-\(', ' bademoticon ', text)
    
    text = _RE_PRICE.sub("<PRICE>", text)
    text = _RE_TIME.sub("<TIME>", text)
    text = _RE_AIRLINES.sub("", text)
    text = _RE_AIRCRAFT.sub("<NUM>", text)
    text = _RE_NUMBER.sub("<NUM>", text)
    
    # Numeri fusi
    text = re.sub(r'\b\w*\d\w*\b', '<NUM>', text)
    
    text = _RE_MULTI_PUNCT.sub(_reduce_punct, text)
    text = _RE_EXTRA_SPACE.sub(" ", text)
    
    # a. Convertire $<NUM> in <PRICE> (es. "$50" diventato "$<NUM>" ora diventa "<PRICE>")
    text = text.replace("$<NUM>", "<PRICE>")
    
    # b. Eliminare il simbolo dell'hashtag mantenendo la parola (#delayed -> delayed)
    text = text.replace("#", "")
    
    # c. Trasformare le chiocciole isolate in "at" (see you @ jfk -> see you at jfk)
    text = re.sub(r'(?:\s|^)@(?:\s|$)', ' at ', text)
    
    # d. Pulire le entità HTML residue (< e > sfuggite al parser base)
    text = text.replace("&lt;", " ").replace("&gt;", " ")
    
    # e. Spazzata finale di sicurezza: Manteniamo solo lettere, numeri, punteggiatura base e i < > dei nostri Token
    text = re.sub(r'[^a-z0-9\s.,!?\'"<>-]', '', text)
    # ---------------------------------------------

    # 12. Collapse multiple spaces into one (ultimo step fondamentale dopo le pulizie)
    text = _RE_EXTRA_SPACE.sub(" ", text)
    
    return text.strip()

# ── 5. APPLY & EXPORT ────────────────────────────────────────────────────────
print("Inizio pulizia di 14.640 tweet... (potrebbe volerci qualche secondo)")
df["tweet_text"] = df["tweet_text"].apply(clean_tweet)

# Rimuoviamo eventuali righe che dopo la pulizia sono diventate completamente vuote
df = df[df["tweet_text"] != ""]

print("\nPulizia completata. Anteprima dei primi 5 tweet:")
for i, tweet in enumerate(df["tweet_text"].head(5)):
    print(f"  [{i}] {tweet}")

output_file = "kaggle_dataset_clean.csv"
df.to_csv(output_file, index=False)

print(f"\nDataset salvato in '{output_file}' ({df.shape[0]} righe).")