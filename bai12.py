# '''Bag of Words Model'''
# from sklearn.feature_extraction.text import CountVectorizer

# Sentences = ['We are using the Bag of Word model', 'Bag of Word model is used for extracting the features.']

# vectorizer_count = CountVectorizer()

# features_text = vectorizer_count.fit_transform(Sentences).todense() # Chuyển Sentences thành matrix với row tương ứng 1 câu, mỗi col tương ứng 1 từ

# print(vectorizer_count.vocabulary_) # Print tập từ vựng mà CountVectorizer đã build dự trên input data. Các từ được gán với 1 số nguyên


# '''Category Prediction'''
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer

# category_map = {
#     'talk.religion.misc': 'Religion',
#     'rec.autos': 'Autos',
#     'rec.sport.hockey': 'Hockey',
#     'sci.electronics': 'Electronics',
#     'sci.space': 'Space'
# }

# # Lấy 1 phần data trong 20 Newsgroups với hàm fetch_20newsgroups. Tập data này chỉ gồm các văn bản trong category_map
# training_data = fetch_20newsgroups( 
#     subset='train',
#     categories=category_map.keys(),
#     shuffle=True,
#     random_state=5
# )

# vectorizer_count = CountVectorizer() # Đối tượng CountVectorizer được sử dụng để chuyển đổi văn bản thành biểu diễn dựa trên tần suất xuất hiện của các từ.
# train_tc = vectorizer_count.fit_transform(training_data.data) # Matrix thể hiện tần suất appear của các từ
# print("\nDimensions of training data:", train_tc.shape)

# tfidf = TfidfTransformer() # Tạo TfidfTransformer để chuyển matrix -> Term Frequency-Inverse Document Frequency
# train_tfidf = tfidf.fit_transform(train_tc) # Train tạo matrix 

# input_data = [
#     'Discovery was a space shuttle',
#     'Hindu, Christian, Sikh all are religions',
#     'We must drive safely',
#     'Puck is a disk made of rubber',
#     'Television, Microwave, Refrigerator all use electricity'
# ]

# classifier = MultinomialNB().fit(train_tfidf, training_data.target) # Mô hình Naive Bayes đã train trên data TF-IDF

# input_tc = vectorizer_count.transform(input_data) # Matrix thể hiện tần suất appear của các từ
# input_tfidf = tfidf.transform(input_tc)

# predictions = classifier.predict(input_tfidf) # Danh sách các dự đoán về danh mục tương ứng cho các đoạn văn bản đầu vào.

# # Vòng lặp print prediction cho mỗi đoạn input văn bản và nội dung của văn bản
# for sent, category_idx in zip(input_data, predictions):
#     category = training_data.target_names[category_idx]
#     print('\nInput Data:', sent, '\nCategory:', category_map[category])


# '''Gender Finder'''
# import random

# from nltk import NaiveBayesClassifier
# from nltk.classify import accuracy as nltk_accuracy
# from nltk.corpus import names

# # Tạo hàm nhận 1 từ và tham số N=2. Dùng để lấy N ký tự cuối cùng và trả về từ điển dạng feature': last_n_letters
# def extract_features(word, N=2):
#     last_n_letters = word[-N:]
#     return {'feature': last_n_letters.lower()}

# if __name__ == '__main__':
#     male_list = [(name, 'male') for name in names.words('male.txt')] # Tạo danh sách các cặp (tên, 'male') từ danh sách tên người đàn ông trong tài nguyên names.words('male.txt') 
#     female_list = [(name, 'female') for name in names.words('female.txt')] # Tương tự male
#     data = male_list + female_list # Tạo list data gồm male và female

#     random.seed(5) # Gen ngẫu nhiên seed để đảm bảo random result có thể regen
#     random.shuffle(data) # Xáo trộn data để đảm bảo random

#     namesInput = ['Rajesh', 'Gaurav', 'Swati', 'Shubha']

#     train_sample = int(0.8 * len(data)) # Lấy 80% để train

#     for i in range(1, 6):
#         print('\nNumber of end letters:', i)
#         features = [(extract_features(n, i), gender) for (n, gender) in data] # Tạo list features bằng cách lấy từ tên và gán gender cho từng tên trong data
#         train_data, test_data = features[:train_sample], features[train_sample:] # Tách data thành tập train và tập test dựa trên train_sample
#         classifier = NaiveBayesClassifier.train(train_data) # Train

#         accuracy_classifier = round(100 * nltk_accuracy(classifier, test_data), 2) # Tính độ chính xác của mô hình trên tập test và lưu vào accuracy_classifier
#         print('Accuracy = ' + str(accuracy_classifier) + '%')

#         for name in namesInput:
#             print(name, '==>', classifier.classify(extract_features(name, i))) # Dự đoán gender dựa trên mô hình đã trained
