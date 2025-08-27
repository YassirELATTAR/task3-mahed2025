import re
from emoji import demojize
from nltk.tokenize import TweetTokenizer


# --- Arabic Tweet Normalizer (Adapted from Mahed script) ---
# NLTK's TweetTokenizer might not be ideal for formal Arabic,
# but it's good for tweet-like short texts with emojis/URLs.

class ArabicTweetNormalizer:
    def __init__(self):
        self.tokenizer = TweetTokenizer()

    def normalize_token(self, token):
        lowercased_token = token.lower()
        if token.startswith("@"):
            return "@USER"
        elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
            return "HTTPURL"
        elif len(token) == 1:
            return demojize(token)
        else:
            if token == "’":
                return "'"
            elif token == "…":
                return "..."
            # Add Arabic-specific normalizations if needed, e.g., for Alif variants, Ta Marbuta etc.
            # For this context, the general TweetTokenizer should handle basic tokenization.
            # More advanced Arabic normalization would involve libraries like PyArabic.
            return token

    def normalize_tweet(self, tweet):
        # Basic Arabic text cleaning (can be expanded)
        tweet = re.sub(r'[إأٱآا]', 'ا', tweet) # Normalize Alif forms
        tweet = re.sub(r'ى', 'ي', tweet)      # Normalize Alef Maqsura to Ya
        tweet = re.sub(r'ؤ', 'ء', tweet)      # Normalize Waw with Hamza to Hamza
        tweet = re.sub(r'ئ', 'ء', tweet)      # Normalize Ya with Hamza to Hamza
        tweet = re.sub(r'ة', 'ه', tweet)      # Normalize Ta Marbuta to Ha (context dependent, but common)
        tweet = re.sub(r'[ًٌٍَُِّْ]', '', tweet) # Remove diacritics

        tokens = self.tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
        norm_tweet = " ".join([self.normalize_token(token) for token in tokens])

        # Apply the same English-centric replacements from the original script
        norm_tweet = (
            norm_tweet.replace("cannot ", "can not ")
            .replace("n't ", " n't ")
            .replace("n 't ", " n't ")
            .replace("ca n't", "can't")
            .replace("ai n't", "ain't")
        )
        norm_tweet = (
            norm_tweet.replace("'m ", " 'm ")
            .replace("'re ", " 're ")
            .replace("'s ", " 's ")
            .replace("'ll ", " 'll ")
            .replace("'d ", " 'd ")
            .replace("'ve ", " 've ")
        )
        norm_tweet = (
            norm_tweet.replace(" p . m .", " p.m.")
            .replace(" p . m ", " p.m ")
            .replace(" a . m .", " a.m.")
            .replace(" a . m ", " a.m ")
        )
        
        # Remove extra spaces
        return " ".join(norm_tweet.split())
