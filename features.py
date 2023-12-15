from textblob import TextBlob
import readability
# import pronouncing

def word_count(lyrics):
    return len(lyrics.split())

def unique_word_count(lyrics):
    return len(set(lyrics.split()))

def sentiment_analysis(lyrics):
    blob = TextBlob(lyrics)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def profanity_count(lyrics):
    profane_words_set = {
    "fuck", "shit", "bitch", "ass", "damn", "bastard", 
    "hell", "whore", "slut", "dick", "pussy", "cock", 
    "nigga", "nigger", "fag", "faggot", "cunt", "crap", 
    "asshole", "motherfucker", "piss", "douche"}

    return sum(1 for word in lyrics.split() if word in profane_words_set)

def readability_scores(lyrics):
    results = readability.getmeasures(lyrics, lang='en')
    return results['readability grades']['FleschReadingEase']
