import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from wordcloud import WordCloud
from wordcloud import STOPWORDS

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")


def grouped_bar(merged_df, metric):

    def replacer(name):
      name = name.replace('S140-test.', '')
      name = name.replace('dublin.', '')
      return name

    tests = ['dublin' if 'dublin' in model else 'test' for model in list(merged_df.index)]
    names = [replacer(name) for name in list(merged_df.index)]

    results = merged_df.copy()
    results['test'] = tests
    results['name'] = names

    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=results, kind="bar",
        x="name", y=metric, hue="test", 
        palette="dark", alpha=.6, height=6)
    g.despine(left=True)
     
    g.set_axis_labels("", metric)
    g.legend.set_title("")

    ax = g.ax
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.show()


def confusion_matrix(y_test, y_pred, classes):
    cm = metrics.confusion_matrix(y_test, y_pred, labels=classes)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=classes)
    disp.plot()
    plt.show()


def wordcloud(train_df):

    df = train_df.copy()

    # Wordcloud with positive tweets
    positive_tweets = df['text'][df["polarity"] == 1]
    stop_words = ["https", "co", "RT"] + list(STOPWORDS)
    positive_wordcloud = WordCloud(max_font_size=50, max_words=50, background_color="white", stopwords = stop_words).generate(str(positive_tweets))
    plt.figure()
    plt.title("Positive Tweets - Wordcloud")
    plt.imshow(positive_wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# Wordcloud with negative tweets
    negative_tweets = df['text'][df["polarity"] == 0]
    stop_words = ["https", "co", "RT"] + list(STOPWORDS)
    negative_wordcloud = WordCloud(max_font_size=50, max_words=50, background_color="white", stopwords = stop_words).generate(str(negative_tweets))
    plt.figure()
    plt.title("Negative Tweets - Wordcloud")
    plt.imshow(negative_wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
