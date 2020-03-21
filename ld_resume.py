import textract
import re
import argparse
import os
import ssl
import nltk
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix
from collections import Counter
import json
import collections
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

def PrintHelp():
    """
    Function to Print out the help section of the code
    """
    usage_text = """Usage: python ld_resume.py [ARGS] 
                  Required Arguments: 
                  --dir </dir/to/folder/>              Path to folder containing resumes 
                  --result-file <output json file>     Path to Output Json File 

                  Optional Arguments: 
                  --min-df <min_df value>              Min DF value for the CountVectorizer
                  --max-df <max_df value>              Max DF value for the CountVectorizer
                  --stop-words <stop words>            Custom Stop Words if any 
                  --num-common-words-ignore <Number>   Number of common words to ignore
                  --num-keywords <Number>              Number of keywords to extract to JSON. default = 5 

                  Example Usage : python ld_resume.py --dir /path/to/folder_containing_resumes/ --result-file /path/output/result.json"""

    print(usage_text)


def create_json_output_file(json_object, file_path):
    """
    Function to serialize an object in a JSON formatted stream and write to a .json file
    Function creates the base directory folders of the final output file, if the directories do not exist.
    If the output file already exists, a new file will be created in its place.
    The indent level of the JSON file is 2, to pretty print.
    input:
        json_object - Object to serialize in JSON format
        file_path - location of the JSON file
    return value : No return value
    """
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(file_path, 'w') as outfile:
        json.dump(json_object, outfile, sort_keys=True, indent=2)

    print(json.dumps(json_object, indent=True, sort_keys=True))
    outfile.close()


def extract_topn_keywords(feature_names, sorted_tf_idf_vector, topn=10):
    """
    Function to get the top N feature names from the tf-idf vector
    The function filters out unigrams/bigrams, based on if a part of was already included in the list.
    :arg
        feature_name - list of feature names extracted from the CountVector
        sorted_tf_idf_vector - Sorted TF-IDF vector created from the TFIDFTransformer
        topn - Number of top N feature names to extract from the vector
    :return
        dict object containing top N feature names and their scores
    """
    result = {}
    for i in (range(len(sorted_tf_idf_vector))):
        if str(feature_names[i]) not in result:
            # print("Checking " + feature_names[sorted_items[i][0]])
            split = str(feature_names[sorted_tf_idf_vector[i][0]]).split(" ")
            if len(split) == 1:
                found = 0
                for entry in result:
                    match = SequenceMatcher(None, split[0], entry).find_longest_match(0, len(split[0]), 0, len(entry))
                    if match.size > 5:
                        found = 1
                if found == 0:
                    result[feature_names[sorted_tf_idf_vector[i][0]]] = sorted_tf_idf_vector[i][1]

            else:
                for word in split:
                    if word in result:
                        result.pop(word, None)
                        result[feature_names[sorted_tf_idf_vector[i][0]]] = sorted_tf_idf_vector[i][1]
                    else:
                        result[feature_names[sorted_tf_idf_vector[i][0]]] = sorted_tf_idf_vector[i][1]

        if len(result) == topn:
            break

    return result


def get_resume_filepaths(input_folder):
    """
    Function to get list of file paths to resumes in pdf format.
        :arg
            Folder to path containing files
        :return
            List of file paths
    """
    list_resumes = []
    for dirpath, _, filenames in os.walk(input_folder):
        for f in filenames:
            if "pdf" in f:
                list_resumes.append(os.path.abspath(os.path.join(dirpath, f)))

    return list_resumes


def MostCommonWords(resume_store, num=20):
    """
    Function to return top N common words in the whole text except the stop words from the nltk library
    This function is used for analyzing if any keywords need to be custom added to the list of stopwords
        :arg
            resume_store - data frame with column "ResumeText" containing resumes and
            num - Number of most common words from the combined corpus of entries under "ResumeText"
        :return
            List of top N common words
    """
    words = []
    nltk_stop_words = set(stopwords.words("english"))
    for resume_words in resume_store["ResumeText"]:
        text = resume_words
        # Remove anything which is not except hypen
        text = re.sub('[^a-zA-Z0-9|^\-]', ' ', text)

        # Remove words with digits
        text = re.sub("\S*\d\S*", "", text).strip()

        # Remove empty hyphens
        text = re.sub(' - ', ' ', text)

        text = text.split()

        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word).lower() for word in text if not word in nltk_stop_words]
        words.extend(text)

    freq = pd.Series(words).value_counts()[:num]

    freq.plot(kind="barh")
    counter = Counter(words)
    common_n = [str(tup[0]).lower() for tup in counter.most_common(num)]
    return common_n


def sort_coo(coo_matrix):
    """
    Function to sort and return vector of coo matrix generated from tf-idf transformer
    :arg
        coo_matrix
    :return
        sorted list of tuples of the sparse coo matrix
    """
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def resume_analyzer(input_folder, output_file, min_df, max_df, stopwords_user=[], num_common_words_in_stopwords=0,
                    num_keywords=5):
    """
        Function to analyze resumes in the input folder and extract N keywords into a JSON file.

    :param input_folder: folder containing resumes
    :param output_file: path to result JSON file
    :param min_df: min_df for the CountVectorizer
    :param max_df: max_df for the CountVectorizer
    :param stopwords_user: Custom Stop words if any provided by the user
    :param num_common_words_in_stopwords: Number of most common words to include in the stop words
    :param num_keywords: Number of keywords to extract in the JSON
    :return:
    """

    list_resumes = get_resume_filepaths(input_folder)
    if len(list_resumes) == 0:
        print("No resumes found in input folder " + input_folder + "\n Exiting program")
        PrintHelp()
        return

    resume_data_store = pd.DataFrame(columns=["PdfFile", "ResumeText"])

    # Extract Text from the pdfs and store in a Pandas Dataframe - resume_data_store
    for resume_path in list_resumes:
        filename_w_ext = os.path.basename(resume_path)
        resume_extract_text = textract.process(resume_path, method='pdfminer', encoding='ascii')
        df = pd.DataFrame([[filename_w_ext, str(resume_extract_text.decode("ASCII"))]],
                          columns=["PdfFile", "ResumeText"])
        resume_data_store = resume_data_store.append(df)

    stop_words = set(stopwords.words("english"))

    # Creating a list of stop words and adding custom stopwords obtained by counting the number of common words in the entire resume text corpus
    if (num_common_words_in_stopwords > 0):
        common_n = MostCommonWords(resume_data_store, num_common_words_in_stopwords)
        stop_words = stop_words.union(common_n)

    # Custom stop words created by analyzing results from various runs.
    new_words_orig = ["india", "http", "software", "greater", "new", "york", "city", "year", "month", "research",
                      "assistant", "things", "college", "maryland", "park", "college park", "deep", "computer", "science",
                      "area", "intern", "data", "science", "january", "february", "march", "april", "may", "june", "july",
                      "august", "september", "october", "november", "december"]

    stop_words = stop_words.union(new_words_orig)

    if (len(stopwords_user) > 0):
        stop_words = stop_words.union(stopwords_user)

    resume_text_collection = []

    for resume_words in resume_data_store["ResumeText"]:
        text = resume_words

        # Remove punctuation marks except hyphen
        text = re.sub('[^a-zA-Z0-9|^\-]', ' ', text)

        # Remove words containing numbers
        text = re.sub("\S*\d\S*", "", text).strip()

        text = text.split()

        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if not word in stop_words]
        text = " ".join(text)
        resume_text_collection.append(text)

    # token_pattern - r"(?u)\b\w[\w-]*\w\b" to respect hyphenated words
    cv = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=stop_words, ngram_range=(1, 2), analyzer='word',
                         token_pattern=r"(?u)\b\w[\w-]*\w\b")
    X = cv.fit_transform(resume_text_collection)

    feature_names = cv.get_feature_names()

    result_json = {}
    list_entries = []
    for i in range(len(resume_data_store)):
        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(X)
        tf_idf_vector = tfidf_transformer.transform(cv.transform([resume_text_collection[i]]))
        sorted_tf_idf_vector = sort_coo(tf_idf_vector.tocoo())
        # extract only the top n; n here is 10
        keywords = extract_topn_keywords(feature_names, sorted_tf_idf_vector, num_keywords)

        key_words_resume = []
        resume = {}
        for k in keywords:
            key_words_resume.append(str(k).title())

        resume["FileName"] = resume_data_store.iloc[i, 0]
        resume["Keywords"] = key_words_resume

        list_entries.append(resume)

    result_json["Resumes"] = list_entries

    create_json_output_file(result_json, output_file)

if __name__ == "__main__":
    # Parsing required and optional arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', dest='dir', required=False, help='<Required> path to folder containing resumes')
    parser.add_argument('--result-file', dest='result_file', required=False,
                        help='<Required> path to the result json file')
    parser.add_argument('--min-df', dest='cv_min_df', required=False, default=0.2, type=int,
                        help='<Optional> Minimum Document Frequency (min_df) value for the CountVectorizer')
    parser.add_argument('--max-df', dest='cv_max_df', required=False, default=0.8, type=int,
                        help='<Optional> Maximum Document Frequency (min_df) value for the CountVectorizer')
    parser.add_argument('--stop-words', dest='stop_words', required=False, default=[], nargs='+',
                        help='<Optional> Custom Stopwords to include in the analyses if any')
    parser.add_argument('--num-common-words-ignore', dest='num_common_words_in_stopwords', required=False, default=0,
                        type=int, help='<Optional> Number of most common words to include in the stop words')
    parser.add_argument('--num-keywords', dest='num_keywords', required=False, default=5, type=int,
                        help='<Optional> Number of keywords to extract from files')

    parser.add_argument("--print-help", dest='print_help', action='store_true')

    parsed_args = parser.parse_args();

    if parsed_args.print_help:
        PrintHelp()
        exit()

    folder = parsed_args.dir
    result_json = parsed_args.result_file

    cv_min_df = parsed_args.cv_min_df
    cv_max_df = parsed_args.cv_max_df
    stopwords_user = parsed_args.stop_words
    num_common_words_in_stopwords = parsed_args.num_common_words_in_stopwords
    num_keywords = parsed_args.num_keywords

    if (str(folder) == ""):
        print("Usage : --dir argument empty")
        PrintHelp()

    if (str(result_json) == ""):
        print("Usage : --result-file argument empty")
        PrintHelp()

    resume_analyzer(folder, result_json, cv_min_df, cv_max_df, stopwords_user, num_common_words_in_stopwords,
                    num_keywords)
