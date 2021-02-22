# Tweet Suport Phrase Prediction

Understanding what descriptions lead to certain sentiment in language is important nowadays when reactions to various decisions are made in seconds on social media. The goal of this project is to construct a model for a given sentence and the label sentiment to predict what phrases in the sentence that best support the given sentiment. In natural language processing, we can formulate this task as a question-answering task where sentiment is a question, the tweet is the context, and the support phrase is the answer. We compare several popular neural network models such as the Long-short Term Memory (LSTM), Gated Recurrent Unit (GRU) for this task. Under a word similarity metric, Jaccard score, we are able to achieve 0.55 score with the model.

## Methods and Results 

### Exploratory Data Analysis and Processing

Prior to model building, it is always good to explore the dataset first. From the distribution of the sentiment in the tweets, we can see that tweets that carry netrual sentiment tend to be shorter than those that are positive or negative. We also see that negative sentiment frequently occurs at text length around 10 or 20. The collected tweets hardly go over 35 words. 

![](/figures/sentiment_distribution.png)

Next, we can get a sense of what words that contribute to different sentiments in the tweets. We plot three different wordclouds to show different collections of vocabularies that count towards different types of sentiment. For the positive wordcloud, the main words include "love", "thank", "good". 

![](/figures/pos-wordcloud.png)

Interestingly, as we look at the neutral word cloud, we see that the main words include "today", "going", "want", "need".

![](/figures/neu-wordcloud.png)

Then, we can see from the negative word cloud that people mostly express their negative sentiment through words like "really", "sad", "sorry".

![](/figures/neg-wordcloud.png)

Besides, we can look at two words together at a time, which is known as bigram, to understand which two words go together often in different sentimental tweets. We can see that "mother day" and "happy mother" are strongly associated with positive sentiment. In addition, negative sentiment is mostly associated with "feel like" and "last night". It maybe good to look into those tweets with "last night" to see if most of them come from news. 

![](/figures/pos-bigram.png)

![](/figures/neg-bigram.png)


## Data Source
[Tweet Sentiment Extraction](https://www.kaggle.com/c/tweet-sentiment-extraction)






