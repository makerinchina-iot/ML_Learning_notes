import jieba
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def dictvec():
    """

    :return:None
    """
    dict = DictVectorizer(sparse=False)

    raw_data = [{'city':'北京', 'temp':28},
                {'city':'上海', 'temp':31},
                {'city':'深圳', 'temp':24}]

    data = dict.fit_transform(raw_data)

    print(dict.get_feature_names())
    print(data)


def countvec():
    """

    :return:
    """
    # raw_data = ["life is too short, I like python to save life", "life is too long, I do not like python"]
    raw_data = ["人生苦短，我用Python", "人生漫长，不用Python"]

    tv = CountVectorizer()
    data = tv.fit_transform(raw_data)
    print(data.toarray())
    print(tv.get_feature_names())


def chinesevec():
    """
    Chinese word vec
    :return:
    """
    c1 = ' '.join(list(jieba.cut("人生这么短，我们要开始使用python进行开发")))

    # print(c1)
    c2 = ' '.join(list(jieba.cut("人生路漫漫，还是放弃使用python吧")))
    c3 = ' '.join(list(jieba.cut("在日常生活中，我们还是使用了golang进行开发工作")))

    tv = CountVectorizer()
    data = tv.fit_transform([c1,c2,c3])
    print(data.toarray())
    print(tv.get_feature_names())


def tfidfvec():
    c1 = ' '.join(list(jieba.cut("人生这么短，我们要开始使用python进行开发")))
    c2 = ' '.join(list(jieba.cut("人生路漫漫，还是放弃使用python吧")))
    c3 = ' '.join(list(jieba.cut("在日常生活中，我们还是使用了golang进行开发工作")))

    tf = TfidfVectorizer()

    data = tf.fit_transform([c1,c2,c3])
    print(data.toarray())
    print(tf.get_feature_names())

if __name__ == "__main__":
    # dictvec()
    # countvec()
    # chinesevec()
    tfidfvec()