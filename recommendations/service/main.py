import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from os import getenv
from sqlalchemy import create_engine
from flask import Flask, request
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def generate_dataset():
    con = create_engine(getenv("DATABASE_URI"))
    tourism = pd.read_sql('SELECT t.id as item_id, t.name, d.description, t.category_id as category, t.location_id as city \
                          FROM tourisms t, tourism_details d \
                          WHERE t.id = d.tourism_id', con)
    user_click = pd.read_sql('SELECT user_id, tourism_id as item_id, total_click as clicked \
                             FROM tourism_user_clicks', con)
    con.dispose()

    all_users = user_click['user_id'].unique()
    all_items = tourism['item_id']

    data_all = []
    for user in all_users:
        for item in all_items:
            data_all.append([user, item])

    df_all = pd.DataFrame(data_all, columns=['user_id', 'item_id'])
    df_all = pd.merge(df_all, tourism, on='item_id', how='left')

    df_final = pd.merge(df_all, user_click, how='left', on=['user_id', 'item_id'], suffixes=('', '_user_clicked'))
    df_final['clicked'].fillna(0, inplace=True)

    return df_final

def generate_model(dataset):
    # Split the data into a training set and a validation set
    df_train, df_val = train_test_split(dataset, test_size=0.2, random_state=42)

    # Define the maximum number of words in the texts to keep based on word frequency
    max_words = 500

    # Tokenizers
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(df_train['description'])

    # Convert the texts to sequences
    description_sequences_train = tokenizer.texts_to_sequences(df_train['description'])
    description_sequences_val = tokenizer.texts_to_sequences(df_val['description'])

    # Pad the sequences so they are all the same length
    description_padded_train = pad_sequences(description_sequences_train, maxlen=max_words)
    description_padded_val = pad_sequences(description_sequences_val, maxlen=max_words)

    # Custom Label Encoding for user_id and item_id
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    encoded_user_ids_train = user_encoder.fit_transform(df_train['user_id'])
    encoded_item_ids_train = item_encoder.fit_transform(df_train['item_id'])

    encoded_user_ids_val = user_encoder.transform(df_val['user_id'])
    encoded_item_ids_val = item_encoder.transform(df_val['item_id'])

    labels_train = df_train['clicked']
    labels_val = df_val['clicked']

    # Build the model
    user_input = layers.Input(shape=(1,), name='user')
    item_input = layers.Input(shape=(1,), name='item')
    description_input = layers.Input(shape=(max_words,), name='description')

    user_embedding = layers.Embedding(input_dim=len(user_encoder.classes_), output_dim=50)(user_input)
    item_embedding = layers.Embedding(input_dim=len(item_encoder.classes_), output_dim=50)(item_input)
    description_embedding = layers.Embedding(input_dim=max_words, output_dim=50)(description_input)

    user_embedding = layers.Flatten()(user_embedding)
    item_embedding = layers.Flatten()(item_embedding)
    description_embedding = layers.GlobalAveragePooling1D()(description_embedding)

    concatenated = layers.Concatenate()([user_embedding, item_embedding, description_embedding])

    dense1 = layers.Dense(128, activation='relu')(concatenated)
    dense2 = layers.Dense(64, activation='relu')(dense1)
    out = layers.Dense(1, activation='sigmoid')(dense2)

    model = tf.keras.Model(inputs=[user_input, item_input, description_input], outputs=out)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit([encoded_user_ids_train, encoded_item_ids_train, description_padded_train], labels_train, epochs=10, validation_data=([encoded_user_ids_val, encoded_item_ids_val, description_padded_val], labels_val))

    # Save the model, label encoders, and tokenizers for future use
    model.save('recommendation_model.h5')
    np.save('user_encoder_classes.npy', user_encoder.classes_)
    np.save('item_encoder_classes.npy', item_encoder.classes_)
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

class Recommender(tf.keras.Model):
    def __init__(self, model_path, user_encoder_path, item_encoder_path, tokenizer_path, df, max_words):
        super(Recommender, self).__init__()
        self.model_path = model_path
        self.user_encoder_path = user_encoder_path
        self.item_encoder_path = item_encoder_path
        self.tokenizer_path = tokenizer_path
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
            
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.string)])
    def serving_default(self, new_user_id):
        self.predict(new_user_id)

    def predict(self, new_user_id):
        if new_user_id not in self.user_encoder.classes_:
            print("New user detected. Assigning random existing user for prediction.")
            new_user_id = np.random.choice(self.user_encoder.classes_)

        all_item_ids = self.df['item_id'].unique().tolist()
        all_descriptions = []

        for item_id in all_item_ids:
            item_data = self.df[self.df['item_id'] == item_id].iloc[0]
            all_descriptions.append(item_data['description'])

        encoded_new_user_id = self.user_encoder.transform([new_user_id]*len(all_item_ids))
        encoded_all_item_ids = self.item_encoder.transform(all_item_ids)

        description_sequences = self.tokenizer.texts_to_sequences(all_descriptions)
        description_padded = pad_sequences(description_sequences, maxlen=self.max_words)

        predictions = self.model.predict([encoded_new_user_id, encoded_all_item_ids, description_padded])
        
        top_10_indices = np.argsort(predictions[:, 0])[-10:]

        recomendations = []
        for index in reversed(top_10_indices):
            recomendations.append({str(all_item_ids[index]): str(predictions[index][0])})

        return recomendations

rec = Recommender('recommendation_model.h5', 'user_encoder_classes.npy', 'item_encoder_classes.npy', 'tokenizer.pickle', generate_dataset(), 500)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def get_tourism_recomendations():
    user_id = request.json.get("user_id")
    recomendations = rec.predict(user_id)
    return {"data": recomendations}, 200

@app.route("/reload", methods=["POST"])
def reload_model():
    dataset = generate_dataset()
    generate_model(dataset)
    rec.reload_model()
    return {"message": "success"}, 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)