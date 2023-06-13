from flask import Flask, request
import json
from bs4 import BeautifulSoup
from selenium.webdriver import Firefox, FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re
import numpy as np
from keras.models import load_model
from transformers import TFBertModel, BertTokenizer


class ReviewScraper:
    def __init__(self):
        # Set up Firefox options
        options = FirefoxOptions()
        options.add_argument("-headless")
        options.set_preference("intl.accept_languages", "id-ID")

        # Initialize WebDriver
        self.driver = Firefox(options=options)

    def scrape(self, url):
        self.articles = {}

        self.driver.get(url)

        review_btn = WebDriverWait(self.driver, 5).until(
            EC.presence_of_element_located((By.CLASS_NAME, "hqzQac"))
        )

        # Close cookie popup
        try:
            self.driver.find_element(By.CLASS_NAME, "L2AGLb").click()
        except:
            pass

        review_btn.click()

        scrollable_div = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "review-dialog-list"))
        )

        for x in range(10):
            self.driver.execute_script(
                "arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div
            )
            time.sleep(1)

        soup = BeautifulSoup(self.driver.page_source, "lxml")

        for item in soup.select(".WMbnJf"):
            div_isi = item.select_one(".review-full-text")
            if not div_isi:
                div_isi = item.select_one(".Jtu6Td span span")
            nama = item.select_one(".TSUbDb").text
            gambar = item.select_one(".lDY1rd").get("src")
            like = item.select_one(".QWOdjf").text
            rating = self.extract_first_number(
                item.select_one(".lTi8oc").get("aria-label")
            )
            isi = div_isi.text
            if nama not in self.articles:
                self.articles[nama] = {
                    "isi": isi,
                    "rating": rating,
                    "gambar": gambar,
                    "like": like,
                }

        return self.articles

    @staticmethod
    def extract_first_number(sentence):
        pattern = r"(\d+(\.\d+)?)"
        match = re.search(pattern, sentence)
        if match:
            number = float(match.group(0))
            return number
        return None


class SentimentAnalyzer:
    def __init__(self, model_path, max_len=200):
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

        # Load the trained model
        self.model = load_model(model_path, custom_objects={"TFBertModel": TFBertModel})

        self.max_len = max_len

    def predict_sentiment(self, text):
        encoded_text = self.encode_reviews(self.tokenizer, [text], self.max_len)
        prediction = self.model.predict(encoded_text)
        sentiment = np.argmax(prediction, axis=-1)[0]
        return int(sentiment)

    @staticmethod
    def encode_reviews(tokenizer, reviews, max_length):
        token_ids = np.zeros(shape=(len(reviews), max_length), dtype=np.int32)
        for i, review in enumerate(reviews):
            encoded = tokenizer.encode(
                review, max_length=max_length, truncation=True, padding="max_length"
            )
            token_ids[i] = encoded
        attention_mask = (token_ids != 0).astype(np.int32)
        return {"input_ids": token_ids, "attention_mask": attention_mask}


# Load the model
analyzer = SentimentAnalyzer("model_sentiment.h5")
# Scrape reviews
scraper = ReviewScraper()
# Flask
app = Flask(__name__)

def analyze(self, words):
    try:
        search = words.replace("", "+")
        reviews = scraper.scrape(
            f"https://www.google.com/search?client=firefox-b-d&q={search}"
        )
        # Add sentiment predictions to reviews
        for name, review in reviews.items():
            sentiment = analyzer.predict_sentiment(review['isi'])
            review["sentimen"] = sentiment
    except:
        return None

    return self.reviews.items()

@app.route('/analyze_sentiment')
def get_sentiment():
    words = request.args.get("words")
    result = analyze(words)

    if not result:
        return {"message": "Location not available"}, 404

    return {"data": list(result.get_result())}, 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)