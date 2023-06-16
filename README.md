<h1 align="center">
  <br>
    <img src="assets/traversee-logo.png" alt="Traversee" width="200">
  <br>
    Traversee ID
  <br>
    <small style="font-size: 16px"><em>Discover The Unforgettables</em></small>
</h1>

<!-- Table Of Contents -->

## Table Of Contents

- [Table Of Contents](#table-of-contents)
- [Overview](#overview)
- [Feature](#feature)
- [Dataset](#dataset)
- [Machine Learning Team](#machine-learning-team)

## Overview

In this project, ML teams will work on implementing machine learning algorithms to provide tourism place recommendation and sentiment analysis. We will develop the models using TensorFlow and may employ transfer learning to leverage pre-existing knowledge. Our models will be deployed using TensorFlow Lite, ensuring efficient performance on mobile devices.

## Feature

### Recommendation System

Our recommendation system employs a hybrid model architecture that combines collaborative filtering and textual feature analysis to provide personalized recommendations. The model utilizes user and item embeddings to capture collaborative signals, allowing it to understand user preferences and item characteristics. Additionally, textual embeddings are generated for features such as descriptions, categories, and other attributes, enabling the model to capture content-based information.

These embeddings are concatenated and passed through dense layers, which learn high-level representations that capture both collaborative and content-based aspects of the data. By combining these different types of embeddings, the model can leverage both user-item interactions and textual information to make accurate recommendations. The model is trained using binary cross-entropy loss and the Adam optimizer, optimizing its ability to predict user-item interactions. Tokenization and padding techniques are applied to textual inputs using the `Tokenizer` and `pad_sequences` from the Keras library, ensuring consistent input dimensions for effective model training and prediction. This hybrid approach enables the recommendation system to provide more accurate and relevant recommendations to users based on both collaborative and content-based information.

### Sentiment Analysis

Our model incorporates a sentiment classification feature that focuses on analyzing the sentiment of tourism place reviews. This feature enables us to provide valuable sentiment information about the reviews, allowing users to gain insights into the overall sentiment and opinions expressed by reviewers regarding different tourism destinations. By leveraging the power of BERT and fine-tuning it on Indonesian text data, our model can accurately classify the sentiment of reviews into different categories, such as positive, negative, or neutral. This sentiment information enhances the recommendation system by enabling users to make more informed decisions based on the sentiment conveyed in the reviews of various tourism places.

The model used is a sentiment classification model based on BERT (Bidirectional Encoder Representations from Transformers), specifically the `indobenchmark/indobert-base-p1` variant pretrained on Indonesian text. It tokenizes input text reviews and passes them through the BERT model to obtain contextualized representations. Dropout layers and dense layers with ReLU activation are employed to prevent overfitting and learn higher-level features from the BERT representations. The final dense layer with softmax activation produces probabilities for sentiment classes. The model is trained using Sparse Categorical Crossentropy loss and Adam optimizer, iterating for 10 epochs with a batch size of 16. The trained model is then saved in `.h5` format for future usage and deployment.

## Dataset

The data used for sentiment analysis in our system is obtained through web scraping from Google reviews. The reviews are then manually labeled for sentiment analysis purposes. The training dataset consists of a total of 4,030 rows of reviews collected from various tourist destinations.

For the recommendation system, we query Wikidata to obtain information about 461 tourist destinations in Indonesia. In addition, we incorporate data from a separate dataset available on Kaggle, which provides information on 438 additional tourism destinations in Indonesia (dataset: https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination).

## Machine Learning Team

| Name                       | Student ID  | Contact                                                                                                                                                                                                                                                                                                           |
| -------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Marcella Komunita Pasaribu | M166DSY1856 | <a href="https://www.linkedin.com/in/marcellakomunita/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white" /></a> <a href="https://github.com/marcellakomunita"><img src="https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white" /></a> |
| Aulia Nur Fadhilah         | M181DSY0386 | <a href="https://www.linkedin.com/in/auliaanf/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white" /></a> <a href="https://github.com/auliaanf"><img src="https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white" /></a>                 |
| Muhammad Rivan Febrian     | M042DSX2824 | <a href="https://www.linkedin.com/in/rivanfebrian123/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white" /></a> <a href="https://github.com/rivanfebrian123"><img src="https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white" /></a>   |
