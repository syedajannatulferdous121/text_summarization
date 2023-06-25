import argparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from heapq import nlargest
from gensim.summarization import keywords

def text_summarization(text, num_sentences=3, keyword_extraction=False):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Create a set of stopwords
    stop_words = set(stopwords.words("english"))
    
    # Create a frequency dictionary for each word
    word_frequencies = {}
    for word in word_tokenize(text.lower()):
        if word not in stop_words:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    
    # Calculate the weighted frequency of each sentence
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_frequencies[word]
                else:
                    sentence_scores[sentence] += word_frequencies[word]
    
    # Get the top N sentences with highest scores
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    
    # Join the summary sentences into a summary text
    summary = " ".join(summary_sentences)
    
    if keyword_extraction:
        # Extract keywords from the text
        extracted_keywords = keywords(text, split=True)
        keywords_summary = ", ".join(extracted_keywords)
        
        return summary, keywords_summary
    
    return summary

# Parse command line arguments
parser = argparse.ArgumentParser(description='Text Summarization')
parser.add_argument('text', type=str, help='The text to summarize')
parser.add_argument('--num_sentences', type=int, default=3, help='Number of sentences in the summary (default: 3)')
parser.add_argument('--keyword_extraction', action='store_true', help='Enable keyword extraction')
args = parser.parse_args()

# Perform text summarization
summary = text_summarization(args.text, num_sentences=args.num_sentences, keyword_extraction=args.keyword_extraction)

# Print the summary and keywords
print("Summary:", summary[0])
if args.keyword_extraction:
    print("Keywords:", summary[1])
