# '''Tokenization'''
# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize

# nltk.download('punkt')

# text = "This is an example sentence. Tokenization is important in NLP."

# # Tokenization thành từng từ (word)
# words = word_tokenize(text)
# print("Word tokens:", words)

# # Tokenization thành từng câu (sentence)
# sentences = sent_tokenize(text)
# print("Sentence tokens:", sentences)

# '''Stemming''' 
# import nltk
# from nltk.stem.porter import PorterStemmer # Là một trong những stemmer đầu tiên được phát triển. Thường có kết quả khá đáng tin cậy, nhưng có thể không giữ nguyên ý nghĩa của từ một cách chính xác. Có thể loại bỏ một số phần của từ một cách quá mức.
# from nltk.stem.lancaster import LancasterStemmer # Là một stemmer được thiết kế để cắt bỏ nhiều hơn so với Porter stemmer. Có thể cắt bỏ nhiều tiền tố và hậu tố hơn so với các stemmer khác. Tương tự như Porter, có thể làm mất đi ý nghĩa của từ trong một số trường hợp.
# from nltk.stem.snowball import SnowballStemmer # Snowball (hay còn gọi là Porter2) là một tập hợp các stemmer được phát triển bởi Martin Porter (người đã phát triển Porter Stemmer). Snowball stemmer cải thiện Porter Stemmer và có thêm hỗ trợ cho nhiều ngôn ngữ. Tập trung vào việc cân nhắc giữa việc cắt bỏ một phần và giữ lại ý nghĩa của từ. Có hiệu suất tốt hơn trong một số trường hợp so với Porter và Lancaster stemmer.

# # Ví dụ văn bản đầu vào
# words = ["running", "jumps", "jumping", "happily", "swimming", "swims"]

# # Sử dụng Porter Stemmer
# porter = PorterStemmer()
# porter_stems = [porter.stem(word) for word in words]
# print("Porter stems:", porter_stems)

# # Sử dụng Lancaster Stemmer
# lancaster = LancasterStemmer()
# lancaster_stems = [lancaster.stem(word) for word in words]
# print("Lancaster stems:", lancaster_stems)

# # Sử dụng Snowball Stemmer (ví dụ tiếng Anh)
# snowball = SnowballStemmer("english")
# snowball_stems = [snowball.stem(word) for word in words]
# print("Snowball stems:", snowball_stems)


# '''Lemmatization'''
# import nltk
# from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')  # Download necessary resource

# # Ví dụ văn bản đầu vào
# words = ["running", "jumps", "jumping", "happily", "swimming", "swims"]

# # Khởi tạo Lemmatizer
# lemmatizer = WordNetLemmatizer()

# # Lemmatization
# lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
# print("Lemmatized words:", lemmatized_words)




'''Chunking'''
import nltk

sentence=[("a","DT"),("clever","JJ"),("fox","NN"),("was","VBP"),
          ("jumping","VBP"),("over","IN"),("the","DT"),("wall","NN")]

grammar = "NP:{<DT>?<JJ>*<NN>}"

parser_chunking = nltk.RegexpParser(grammar)

parser_chunking.parse(sentence)

Output_chunk = parser_chunking.parse(sentence)

Output_chunk.draw()