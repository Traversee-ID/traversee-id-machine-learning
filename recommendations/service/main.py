import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import random
from flask import Flask, request
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_dataset(file_name):
    df = pd.read_csv(file_name)
    df['item_id'] = [str(i) for i in range(len(df))]

    df = df[['item_id', 'description', 'category', 'city']]

    json_data = [
        {
            "user_id": "user1",
            "clickedItems": random.choices(df['item_id'], k=100)
        },
        {
            "user_id": "user2",
            "clickedItems": random.choices(df['item_id'], k=100)
        },
        {
            "user_id": "user3",
            "clickedItems": random.choices(df['item_id'], k=100)
        },
        {
            "user_id": "user4",
            "clickedItems": random.choices(df['item_id'], k=100)
        },
        {
            "user_id": "user5",
            "clickedItems": random.choices(df['item_id'], k=100)
        }
    ]

    data = []
    for user in json_data:
        for item in user['clickedItems']:
            data.append([user['user_id'], item, 1])
    df_user_clicked = pd.DataFrame(data, columns=['user_id', 'item_id', 'clicked'])

    all_users = df_user_clicked['user_id'].unique()
    all_items = df['item_id']

    data_all = []
    for user in all_users:
        for item in all_items:
            data_all.append([user, item])

    df_all = pd.DataFrame(data_all, columns=['user_id', 'item_id'])
    df_all = pd.merge(df_all, df, on='item_id', how='left')

    df_final = pd.merge(df_all, df_user_clicked, how='left', on=['user_id', 'item_id'], suffixes=('', '_user_clicked'))
    df_final['clicked'].fillna(0, inplace=True)

    return df_final

class Recommender:
    def __init__(self, model_path, user_encoder_path, item_encoder_path, tokenizer_path, tokenizer_categories_path, tokenizer_other_path, df, max_words):
        self.model_path = model_path
        self.user_encoder_path = user_encoder_path
        self.item_encoder_path = item_encoder_path
        self.tokenizer_path = tokenizer_path
        self.tokenizer_categories_path = tokenizer_categories_path
        self.tokenizer_other_path = tokenizer_other_path
        self.df = df
        self.max_words = max_words
        self.reload_model()

    def reload_model(self):
        self.model = tf.keras.models.load_model(self.model_path)

        self.user_encoder = LabelEncoder()
        self.user_encoder.classes_ = np.load(self.user_encoder_path, allow_pickle=True)

        self.item_encoder = LabelEncoder()
        self.item_encoder.classes_ = np.load(self.item_encoder_path, allow_pickle=True)

        with open(self.tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        with open(self.tokenizer_categories_path, 'rb') as handle:
            self.tokenizer_categories = pickle.load(handle)

        with open(self.tokenizer_other_path, 'rb') as handle:
            self.tokenizer_other = pickle.load(handle)

    def predict(self, new_user_id):
        all_item_ids = self.df['item_id'].unique().tolist()
        all_categories = []
        all_descriptions = []
        all_other_attributes = []

        for item_id in all_item_ids:
            item_data = self.df[self.df['item_id'] == item_id].iloc[0]
            all_categories.append(item_data['category'])
            all_descriptions.append(item_data['description'])
            all_other_attributes.append(item_data['city'])

        if new_user_id not in self.user_encoder.classes_:
            print("New user detected. Assigning random existing user for prediction.")
            new_user_id = np.random.choice(self.user_encoder.classes_)

        encoded_new_user_id = self.user_encoder.transform([new_user_id]*len(all_item_ids))
        encoded_all_item_ids = self.item_encoder.transform(all_item_ids)

        description_sequences = self.tokenizer.texts_to_sequences(all_descriptions)
        description_padded = pad_sequences(description_sequences, maxlen=self.max_words)

        category_sequences = self.tokenizer_categories.texts_to_sequences(all_categories)
        category_padded = pad_sequences(category_sequences, maxlen=self.max_words)

        other_sequences = self.tokenizer_other.texts_to_sequences(all_other_attributes)
        other_padded = pad_sequences(other_sequences, maxlen=self.max_words)

        predictions = self.model.predict([encoded_new_user_id, encoded_all_item_ids, description_padded, category_padded, other_padded])
        top_10_indices = np.argsort(predictions[:, 0])[-10:]

        recomendations = []
        for index in reversed(top_10_indices):
            recomendations.append({str(all_item_ids[index]): str(predictions[index][0])})

        return recomendations

rec = Recommender('recommendation_model.h5', 'user_encoder_classes.npy', 'item_encoder_classes.npy', 'tokenizer.pickle', 'tokenizer_categories.pickle', 'tokenizer_other.pickle', generate_dataset('final.csv'), 500)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def get_tourism_recomendations():
    user_id = request.json.get("user_id")
    recomendations = rec.predict(user_id)
    return {"data": recomendations}, 200

@app.route("/reload", methods=["POST"])
def reload_model():
    rec.reload_model()
    return {"message": "success"}, 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)