# CS421: Natural Language Processing
# University of Illinois at Chicago
# Chatbot Project - Giuseppe Stracquadanio

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import sklearn
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import string
import re
import csv
import nltk


# the provided w2v.pkl file which contains the pre-trained word2vec representations
EMBEDDING_FILE = "w2v.pkl"


# Function: load_w2v
# filepath: path of w2v.pkl
# Returns: A dictionary containing words as keys and pre-trained word2vec representations as numpy arrays of shape (300,)
def load_w2v(filepath):
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)


# Function: load_as_list(fname)
# fname: A string indicating a filename
# Returns: Two lists: one a list of document strings, and the other a list of integers
#
# This helper function reads in the specified, specially-formatted CSV file
# and returns a list of documents (documents) and a list of binary values (label).
def load_as_list(fname):
    df = pd.read_csv(fname)
    documents = df['review'].values.tolist()
    labels = df['label'].values.tolist()
    return documents, labels


# Function: extract_user_info, see project statement for more details
# user_input: A string of arbitrary length
# Returns: name as string
def extract_user_info(user_input):
    name = ""
    name_regex = re.compile(r"(?:^|[\s]+)(([A-Z][a-zA-Z\.\-&']*\s+){1,3})([A-Z][a-zA-Z\.\-&']*)(?:[\s]+|$)")
    res = name_regex.search(user_input)
    if res:
        name = res.group(1,3)
        name = name[0] + name[1]
    return name


# Function to convert a given string into a list of tokens
# Args:
#   inp_str: input string 
# Returns: token list, dtype: list of strings
def get_tokens(inp_str):
    # return inp_str.split()
    # Updating get_tokens function using nltk.tokenize API
    return nltk.tokenize.word_tokenize(inp_str) # language='english' by default

def get_sentences(inp_str):
    return nltk.tokenize.sent_tokenize(inp_str) # language='english' by default

# Function: preprocessing, see project statement for more details
# Args:
#   user_input: A string of arbitrary length
# Returns: A string of arbitrary length
def preprocessing(user_input):
    modified_input = ""
    tokens = get_tokens(user_input)
    filtered_tokens = []
    for token in tokens:
        punct_flag = False
        for punct in string.punctuation:
            if token == punct:
                punct_flag = True
                break
        if not punct_flag:
            filtered_tokens.append(token)
    for token in filtered_tokens:
        modified_input += token + " "
    modified_input = modified_input.lower()
    return modified_input[0:-1]


# Function: vectorize_train, see project statement for more details
# training_documents: A list of strings
# Returns: An initialized TfidfVectorizer model, and a document-term matrix, dtype: scipy.sparse.csr.csr_matrix
def vectorize_train(training_documents):
    # Initialize the TfidfVectorizer model and document-term matrix
    vectorizer = TfidfVectorizer()
    pp_documents = []
    for doc in training_documents:
        pp_documents.append(preprocessing(doc))
    tfidf_train = vectorizer.fit_transform(pp_documents)
    return vectorizer, tfidf_train


# Function: vectorize_test, see project statement for more details
# vectorizer: A trained TFIDF vectorizer
# user_input: A string of arbitrary length
# Returns: A sparse TFIDF representation of the input string of shape (1, X), dtype: scipy.sparse.csr.csr_matrix
#
# This function computes the TFIDF representation of the input string, using
# the provided TfidfVectorizer.
def vectorize_test(vectorizer, user_input):
    tfidf_test = vectorizer.transform([preprocessing(user_input)])
    return tfidf_test


# Function: train_nb_model(training_documents, training_labels)
# training_data: A sparse TfIDF document-term matrix, dtype: scipy.sparse.csr.csr_matrix
# training_labels: A list of integers (0 or 1)
# Returns: A trained model
def train_nb_model(training_data, training_labels):
    # Initialize the GaussianNB model and the output label
    nb_model = GaussianNB()
    # Write your code here.  You will need to make use of the GaussianNB fit()
    # function.  You probably need to transfrom your data into a dense numpy array.
    nb_model.fit(training_data.toarray(), training_labels)
    return nb_model

# Function: get_model_prediction(nb_model, tfidf_test)
# nb_model: A trained GaussianNB model
# tfidf_test: A sparse TFIDF representation of the input string of shape (1, X), dtype: scipy.sparse.csr.csr_matrix
# Returns: A predicted label for the provided test data (int, 0 or 1)
def get_model_prediction(nb_model, tfidf_test):
    pred_label = nb_model.predict(tfidf_test)
    return pred_label


# Function: w2v(word2vec, token)
# word2vec: The pretrained Word2Vec representations as dictionary
# token: A string containing a single token
# Returns: The Word2Vec embedding for that token, as a numpy array of size (300,)
#
# This function provides access to 300-dimensional Word2Vec representations
# pretrained on Google News.  If the specified token does not exist in the
# pretrained model, it should return a zero vector; otherwise, it returns the
# corresponding word vector from the word2vec dictionary.
def w2v(word2vec, token):
    word_vector = np.zeros(300,)
    if token in word2vec:
        word_vector = np.array(word2vec[token]).reshape((300, ))
    return word_vector


# Function: string2vec(word2vec, user_input)
# word2vec: The pretrained Word2Vec model
# user_input: A string of arbitrary length
# Returns: A 300-dimensional averaged Word2Vec embedding for that string
#
# This function preprocesses the input string, tokenizes it using get_tokens, extracts a word embedding for
# each token in the string, and averages across those embeddings to produce a
# single, averaged embedding for the entire input.
def string2vec(word2vec, user_input):
    tokens = get_tokens(preprocessing(user_input))
    string_embeddings = np.zeros((len(tokens), 300))
    for i in range(len(tokens)):
        token = tokens[i]
        token_embedding = w2v(word2vec, token)
        string_embeddings[i] = token_embedding
    embedding = string_embeddings.mean(axis=0).reshape((300, ))
    return embedding


# Function: instantiate_models()
# This function does not take any input
# Returns: Three instantiated machine learning models
#
# This function instantiates the three imported machine learning models, and
# returns them for later downstream use.  You do not need to train the models
# in this function.
def instantiate_models():
    logistic = sklearn.linear_model.LogisticRegression(random_state=100)
    svm = sklearn.svm.LinearSVC(random_state=100)
    mlp = sklearn.neural_network.MLPClassifier(random_state=100)
    return logistic, svm, mlp


# Function: train_model(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# training_data: A list of training documents
# training_labels: A list of integers (all 0 or 1)
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using averaged Word2Vec
# embeddings for the training documents.
def train_model(model, word2vec, training_documents, training_labels):
    train_size = len(training_documents)
    training_embeddings = np.zeros((train_size, 300))
    # Preprocessing (obtaining embeddings) of input training data
    for i, training_document in enumerate(training_documents):
        training_embedding = string2vec(word2vec, training_document)
        training_embeddings[i] = training_embedding
    model.fit(training_embeddings, training_labels)
    return model


# Function: test_model(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# test_data: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  It compares the predicted and actual test labels and returns
# precision, recall, f1, and accuracy scores.
def test_model(model, word2vec, test_documents, test_labels):
    test_size = len(test_documents)
    test_embeddings = np.ones((test_size, 300))
    # Preprocessing (obtaining embeddings) of input training data
    for i, test_document in enumerate(test_documents):
        test_embedding = string2vec(word2vec, test_document)
        test_embeddings[i] = test_embedding

    pred_labels = model.predict(test_embeddings)

    precision = sklearn.metrics.precision_score(test_labels, pred_labels)
    recall = sklearn.metrics.recall_score(test_labels, pred_labels)
    f1 = sklearn.metrics.f1_score(test_labels, pred_labels)
    accuracy = sklearn.metrics.accuracy_score(test_labels, pred_labels)

    return precision, recall, f1, accuracy


# Function: count_words(user_input)
# user_input: A string of arbitrary length
# Returns: An integer value
#
# This function counts the number of words in the input string.
def count_words(user_input):
    tokens = get_tokens(user_input)
    filtered_tokens = []
    for token in tokens:
        punct_flag = False
        for punct in string.punctuation:
            if token == punct:
                punct_flag = True
                break
        if not punct_flag:
            filtered_tokens.append(token)
    return len(filtered_tokens)

# Function: words_per_sentence(user_input)
# user_input: A string of arbitrary length
# Returns: A floating point value
#
# This function computes the average number of words per sentence
def words_per_sentence(user_input):
    sents = get_sentences(user_input)
    tot_count = 0
    for sent in sents:
        tot_count += count_words(sent)
    return tot_count/len(sents)


# Function: get_pos_tags(user_input)
# user_input: A string of arbitrary length
# Returns: A list of (token, POS) tuples
#
# This function tags each token in the user_input with a Part of Speech (POS) tag from Penn Treebank.
def get_pos_tags(user_input):
    tokens = get_tokens(user_input)
    tagged_input = nltk.pos_tag(tokens)
    return tagged_input


# Function: get_pos_categories(tagged_input)
# tagged_input: A list of (token, POS) tuples
# Returns: Seven integers, corresponding to the number of pronouns, personal
#          pronouns, articles, past tense verbs, future tense verbs,
#          prepositions, and negations in the tagged input
#
# This function counts the number of tokens corresponding to each of six POS tag
# groups, and returns those values.  The Penn Treebag tags corresponding that
# belong to each category can be found in Table 2 of the project statement.
def get_pos_categories(tagged_input):
    num_pronouns = 0
    num_prp = 0
    num_articles = 0
    num_past = 0
    num_future = 0
    num_prep = 0

    pos_categories = {
        "Pronouns": ["PRP", "PRP$", "WP", "WP$"],
        "Personal pronouns": ["PRP"],
        "Articles": ["DT"],
        "Paste tense verbs": ["VBD", "VBN"],
        "Future tense verbs": ["MD"],
        "Prepositions": ["IN"]
    }

    for _, pos_tag in tagged_input:
        # Note: The same POS tag (i.e. the same token) can contribute to more than one counter.
        # For example, if we have a token with tag "PRP", we must increase both num_pronouns and num_prp counters.
        if pos_tag in pos_categories["Pronouns"]:
            num_pronouns += 1
        if pos_tag in pos_categories["Personal pronouns"]:
            num_prp += 1
        if pos_tag in pos_categories["Articles"]:
            num_articles += 1
        if pos_tag in pos_categories["Paste tense verbs"]:
            num_past += 1
        if pos_tag in pos_categories["Future tense verbs"]:
            num_future += 1
        if pos_tag in pos_categories["Prepositions"]:
            num_prep += 1

    return num_pronouns, num_prp, num_articles, num_past, num_future, num_prep



# Function: count_negations(user_input)
# user_input: A string of arbitrary length
# Returns: An integer value
#
# This function counts the number of negation terms in a user input string
def count_negations(user_input):
    # List of negations taken from https://www.grammarly.com/blog/negatives/
    negations = ["no", "not", "neither", "nor", "never", "n't", "hardely", "scarcely", "seldom", "rarely", "nothing", "nobody", "nowhere"]
    tokens = get_tokens(user_input)
    num_negations = 0
    for token in tokens:
        for neg in negations:
            if token == neg:
                num_negations += 1
                break
    return num_negations


# Function: summarize_analysis(num_words, wps, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations)
# num_words: An integer value
# wps: A floating point value
# num_pronouns: An integer value
# num_prp: An integer value
# num_articles: An integer value
# num_past: An integer value
# num_future: An integer value
# num_prep: An integer value
# num_negations: An integer value
# Returns: A list of three strings
#
# This function identifies the three most informative linguistic features from
# among the input feature values, and returns the psychological correlates for
# those features.  num_words and/or wps should be included if, and only if,
# their values exceed predetermined thresholds.  The remainder of the three
# most informative features should be filled by the highest-frequency features
# from among num_pronouns, num_prp, num_articles, num_past, num_future,
# num_prep, and num_negations.
def summarize_analysis(num_words, wps, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations):
    informative_correlates = []

    # Creating a reference dictionary with keys = linguistic features, and values = psychological correlates.
    # informative_correlates should hold a subset of three values from this dictionary.
    # DO NOT change these values for autograder to work correctly
    psychological_correlates = {}
    psychological_correlates["num_words"] = "Talkativeness, verbal fluency"
    psychological_correlates["wps"] = "Verbal fluency, cognitive complexity"
    psychological_correlates["num_pronouns"] = "Informal, personal"
    psychological_correlates["num_prp"] = "Personal, social"
    psychological_correlates["num_articles"] = "Use of concrete nouns, interest in objects/things"
    psychological_correlates["num_past"] = "Focused on the past"
    psychological_correlates["num_future"] = "Future and goal-oriented"
    psychological_correlates["num_prep"] = "Education, concern with precision"
    psychological_correlates["num_negations"] = "Inhibition"

    # Set thresholds
    num_words_threshold = 100
    wps_threshold = 20

    if num_words > num_words_threshold:
        informative_correlates.append(psychological_correlates["num_words"])
    if wps > wps_threshold:
        informative_correlates.append(psychological_correlates["wps"])

    # Ordering remaining linguistic features
    feature_dict = {
        "num_pronouns": num_pronouns,
        "num_prp": num_prp,
        "num_articles": num_articles,
        "num_past": num_past,
        "num_future": num_future,
        "num_prep": num_prep,
        "num_negations": num_negations
    }

    tie_breaker = dict(enumerate(feature_dict.keys()))
    tie_breaker = dict((y, x) for x,y in tie_breaker.items())
    feature_dict_sorted = dict(sorted(feature_dict.items(), key=lambda item: (-item[1], tie_breaker[item[0]])))

    for feature_item in feature_dict_sorted.items():
        feature_name, _ = feature_item
        if len(informative_correlates) < 3:
            informative_correlates.append(psychological_correlates[feature_name])
        else:
            break

    return informative_correlates


# -------------------------- New in Project Part 4 --------------------------
# Function: welcome_state
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements the chatbot's welcome states.  Feel free to customize
# the welcome message!  In this state, the chatbot greets the user.
def welcome_state():
    # Display a welcome message to the user
    # *** Replace the line below with your updated welcome message from Project Part 1 ***
    print("Welcome to the CS 421 chatbot!\n")

    return ""


# Function: get_name_state
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that requests the user's name and then
# processes the user's response to extract the name.
def get_name_state():
    # Request the user's name and accept a user response of arbitrary length
    user_input = input("What is your name?\n")

    # Extract the user's name
    name = extract_user_info(user_input)

    # Show name and thank the user
    user_input = print(f"Thanks {name}!")

    return ""


# Function: sentiment_analysis_state
# model: The trained classification model used for predicting sentiment (best one)
# word2vec: The word2vec dictionary
# first_time (bool): indicates whether the state is active for the first time. HINT: use this parameter to determine next state.
# Returns: A string indicating the next state
#
# This function implements a state that asks the user for input and predicts their sentiment
def sentiment_analysis_state(model, word2vec, first_time=False):
    # Check the user's current sentiment
    if first_time == True:
        user_input = input("Welcome! What do you want to talk about today?\n")
    else:
        user_input = input("It's you again! What do you want to talk about?\n")

    # Predict user's sentiment
    w2v_test = string2vec(word2vec, user_input)

    # label = None
    label = model.predict(w2v_test.reshape(1, -1)) # Use this if you select one of the other models (swap mlp for svm, etc.)

    if label == 0:
        print("Hmm, it seems like you're feeling a bit down.")
    elif label == 1:
        print("It sounds like you're in a positive mood!")
    else:
        print("Hmm, that's weird.  My classifier predicted a value of: {0}".format(label))

    return ""


# Function: stylistic_analysis_state
# This function does not take any arguments
# Returns: A string indicating the next state
#
# This function implements a state that asks the user what's on their mind, and
# then analyzes their response to identify informative linguistic correlates to
# psychological status.
def stylistic_analysis_state(first_time=False):
    if first_time == True:
        user_input = input("I'd also like to do a quick stylistic analysis. What's on your mind today?\n")
    else:
        user_input = input("Let's do another quick stylistic analysis. What's on your mind today?\n")

    num_words = count_words(user_input)
    wps = words_per_sentence(user_input)
    pos_tags = get_pos_tags(user_input)
    num_pronouns, num_prp, num_articles, num_past, num_future, num_prep = get_pos_categories(pos_tags)
    num_negations = count_negations(user_input)

    # Uncomment the code below to view your output from each individual function
    # print("num_words:\t{0}\nwps:\t{1}\npos_tags:\t{2}\nnum_pronouns:\t{3}\nnum_prp:\t{4}"
    #      "\nnum_articles:\t{5}\nnum_past:\t{6}\nnum_future:\t{7}\nnum_prep:\t{8}\nnum_negations:\t{9}".format(
    #    num_words, wps, pos_tags, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations))

    # Generate a stylistic analysis of the user's input
    informative_correlates = summarize_analysis(num_words, wps, num_pronouns,
                                                num_prp, num_articles, num_past,
                                                num_future, num_prep, num_negations)
    print("Thanks!  Based on my stylistic analysis, I've identified the following psychological correlates in your response:")
    for correlate in informative_correlates:
        print("- {0}".format(correlate))


    return ""


# Function: check_next_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that checks to see what the user would like
# to do next.  The user should be able to indicate that they would like to quit
# (in which case the state should be "quit"), redo the sentiment analysis
# ("sentiment_analysis"), or redo the stylistic analysis ("stylistic_analysis").
def check_next_state():
    pattern = r"^(?P<abc>[abc])|(?P<sentiment>[sS]entiment [aA]nalysis)|(?P<style>[sS]tylistic [aA]nalysis)|(?P<end>[eE]nd)$"
    prog = re.compile(pattern)
    print("What would you like to do next? [You can either type the option letter or your choice entirely!]")
    print("(a) End the conversation")
    print("(b) Perform again Sentiment Analysis")
    print("(c) Perform again Stylistic Analysis")
    usr_input = input("Type here you choice: ")
    result = prog.match(usr_input)

    # No matching at all, with the user's input
    if result is None:
        return None

    result = result.groupdict()
    if result['abc'] is not None:
        return result['abc']
    elif result['sentiment'] is not None:
        return 'b'
    elif result['style'] is not None:
        return 'c'
    elif result['end'] is not None:
        return 'a'
    return None

# Function: run_chatbot
# model: A trained classification model
# word2vec: The pretrained Word2Vec dictionary (leave empty if not using word2vec based model)
# Returns: This function does not return any values
#
# This function implements the main chatbot system --- it runs different
# dialogue states depending on rules governed by the internal dialogue
# management logic, with each state handling its own input/output and internal
# processing steps.  The dialogue management logic should be implemented as
# follows:
# welcome_state() (IN STATE) -> get_info_state() (OUT STATE)
# get_info_state() (IN STATE) -> sentiment_analysis_state() (OUT STATE)
# sentiment_analysis_state() (IN STATE) -> stylistic_analysis_state() (OUT STATE - First time sentiment_analysis_state() is run)
#                                    check_next_state() (OUT STATE - Subsequent times sentiment_analysis_state() is run)
# stylistic_analysis_state() (IN STATE) -> check_next_state() (OUT STATE)
# check_next_state() (IN STATE) -> sentiment_analysis_state() (OUT STATE option 1) or
#                                  stylistic_analysis_state() (OUT STATE option 2) or
#                                  terminate chatbot
def run_chatbot(model, word2vec):
    welcome_state()
    get_name_state()
    sentiment_analysis_state(model, word2vec, first_time=True)
    stylistic_analysis_state(first_time=True)
    while True:
        next_state = check_next_state()
        if next_state == "b":
            sentiment_analysis_state(model, word2vec)
        elif next_state == 'c':
            stylistic_analysis_state()
        elif next_state == 'a':
            print("Thank you for talking with me! See you next time :)")
            break
        else:
            print("Error! Your input doesn't specify any permissible option.")
            print("Please insert one valid option.")
            continue


# ----------------------------------------------------------------------------

if __name__ == "__main__":

    # Set things up ahead of time by training the TfidfVectorizer and Naive Bayes model
    documents, labels = load_as_list("dataset.csv")

    # Load the Word2Vec representations so that you can make use of it later
    word2vec = load_w2v(EMBEDDING_FILE)

    # Instantiate and train the machine learning models
    logistic, svm, mlp = instantiate_models()
    logistic = train_model(logistic, word2vec, documents, labels)
    svm = train_model(svm, word2vec, documents, labels)
    mlp = train_model(mlp, word2vec, documents, labels)

    # print("Word2Vec embedding for {0}:\t{1}".format("vaccine", w2v(word2vec, "vaccine")))

    # Loading the dataset
    test_documents, test_labels = load_as_list("test.csv")
    models = [logistic, svm, mlp]
    model_names = ["Logistic Regression", "SVM", "Multilayer Perceptron"]
    outfile = open("classification_report.csv", "w", newline='\n')
    outfile_writer = csv.writer(outfile)
    outfile_writer.writerow(["Name", "Precision", "Recall", "F1", "Accuracy"]) # Header row
    i = 0
    while i < len(models): # Loop through other results
        p, r, f, a = test_model(models[i], word2vec, test_documents, test_labels)
        if models[i] == None: # Models will be None if functions have not yet been implemented
            outfile_writer.writerow([model_names[i],"N/A", "N/A", "N/A", "N/A"])
        else:
            outfile_writer.writerow([model_names[i], p, r, f, a])
        i += 1
    outfile.close()

    # For reference, let us also compute the accuracy for the Naive Bayes model
    # vectorizer, tfidf_train = vectorize_train(documents)
    # lexicon = [preprocessing(d) for d in test_documents]
    # tfidf_test = vectorizer.transform(lexicon)
    # naive = train_nb_model(tfidf_train, labels)
    # predictions = naive.predict(tfidf_test.toarray())
    # acc = np.sum(np.array(test_labels) == predictions) / len(test_labels)
    # print("Naive Bayes Accuracy:", acc)

    # Reference code to run the chatbot
    # Replace MLP with your best performing model
    run_chatbot(mlp, word2vec)
