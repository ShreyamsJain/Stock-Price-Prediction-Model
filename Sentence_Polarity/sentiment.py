####################################################################
# Licence:    Creative Commons (see COPYRIGHT)                     #
# Authors:    Nikolaos Pappas, Georgios Katsimpras                 #
#             {nik0spapp, gkatsimpras}@gmail.com                   # 
# Supervisor: Efstathios stamatatos                                #
#             stamatatos@aegean.gr                                 #
# University of the Aegean                                         #
# Department of Information and Communication Systems Engineering  #
# Information Management Track (MSc)                               #
# Karlovasi, Samos                                                 #
# Greece                                                           #
####################################################################

import os
import sys
import nltk
import pickle
from bootstrapping import Bootstrapping
from pos import SequentialTagger
from hp_classifiers import HpObj, HpSubj
from polarity import PolarityClassifier  
from replacer import RepeatReplacer
from terminal_colors import Tcolors
import pandas as pd
import traceback

DEBUG = False

class Sentiment:
    """
        Sentiment: Analyses the global sentiment of given text regions  
        that are decomposed to sentences, using bootstrapping methods for 
        subjectivity and polarity classification. All sub modules except 
        from POS tagging are learning by experience.
    """
    
    def __init__(self):
        self.pos_tagger = SequentialTagger()
        self.hp_obj = HpObj(debug=DEBUG)
        self.hp_subj = HpSubj(debug=DEBUG)
        self.lexicon = self.hp_obj.lexicon
        self.bootstrapping = Bootstrapping(self.hp_obj, self.hp_subj, self.pos_tagger, debug=DEBUG) 
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.total_sentences = ["good","bad"]
        self.total_sentiments = ["positive","negative"]
        
    def analyze(self, clean_text_areas):
        """
            Analysis of text regions using the following order: Each sentence per
            region is passed from the subjectivity classification using bootstrapping
            method and then if it turns out to be subjective it is passed 
            from the polarity classification using bootstrapping method also.
            Finally, it results to a decision for the sentiment of the sentence
            and the overall sentiment of the regions. 
        """ 
        if len(clean_text_areas) > 0:  
            for clean_text in clean_text_areas:
                #print "clean_text is " + clean_text
                # Sentence detection
                clean_text = self.normalize(clean_text)
                try:
                    sentences = self.sentence_tokenizer.tokenize(clean_text)
                except Exception, e:
                    print "Hitting Error"
                    traceback.print_exc()
                    return {}
                sentiments = [] 
                scores = []
                nscores = []
                results = {'positive':{'count' : 0, 'score' : 0, 'nscore' : 0},
                           'neutral':{'count' : 0, 'score' : 0, 'nscore' : 0},
                           'negative':{'count' : 0, 'score' : 0, 'nscore' : 0}}
                
                print
                print Tcolors.ACT + " Checking block of text:"
                for i, sentence in enumerate(sentences):
                    print "[" + str(i+1) + "] " + sentence
                for i, sentence in enumerate(sentences):
                    # Proceed to subjectivity classification (bootstrapping procedure).
                    # (This step could be skipped in case you deal with subjective sentences only.)
                    sentiment = ""
                    previous = ""
                    next = ""
                    score = 0
                    nscore = 0
                    if i == 0 and i + 1 < len(sentences): 
                        next = sentences[i+1] 
                    elif i != 0 and i < len(sentences):
                        if i + 1 != len(sentences):
                            next = sentences[i+1]
                        previous = sentences[i-1] 
                     
                    if DEBUG: print Tcolors.ACT + " Analyzing subjectivity..." 
                    result = self.bootstrapping.classify(sentence, previous, next) 
                    if result is None:
                        res = 'Not found!'
                    else:
                        res = result
                    if DEBUG:
                        print Tcolors.RES + Tcolors.OKGREEN + " " + res + Tcolors.ENDC
                        print
                    
                    # If sentence is subjective 
                    if result == 'subjective' or result is None:
                        # Proceed to polarity classification
                        if DEBUG: print Tcolors.ACT + " Analyzing sentiment..."
                        polarity_classifier = PolarityClassifier(self.pos_tagger, self.lexicon, debug=DEBUG)
                        sentiment, score, nscore = polarity_classifier.classify(sentence)
                        if DEBUG: print Tcolors.RES + Tcolors.OKGREEN + " " + sentiment + Tcolors.ENDC
                    # If sentence is objective
                    elif result == 'objective':
                        sentiment = 'neutral'  
                    
                    # Collect high-confidence training instances for SVM classifier.
                    # After the training, SVM can be used to classify new sentences.
                    #if sentiment != "neutral" and sentiment != "": 
                        #if sentiment != "neutral" and abs(nscore) >= 0.4:
                        #   self.total_sentences.append(sentence)
                        #   self.total_sentiments.append(sentiment)
                        
                    # Store results to memory
                    sentiments.append(sentiment)
                    scores.append(score)
                    nscores.append(nscore)
                    
                    # Update score
                    if results.has_key(sentiment):
                        results[sentiment]['nscore'] += nscore
                        results[sentiment]['score'] += score
                        results[sentiment]['count'] += 1 
                          
                print       
                print Tcolors.ACT + " Overall sentiment analysis:"
                print Tcolors.BGH
                print " Parts: ", len(sentences)
                print " Sentiments: ", sentiments
                print " Scores: ", scores 
                print " Results: ", "},\n\t    ".join((str)(results).split("}, "))
                print Tcolors.C

                pcount = results['positive']['count']
                ncount = results['negative']['count'] 
                total = len(sentences)
                print Tcolors.BG
                subjective = 0
                objective = 0
                subjective = (float)(pcount + ncount)*100 / total
                objective = 100 - ((float)(pcount + ncount)*100 / total)
                print " subjective".ljust(16,"-") + "> %.2f" % ((float)(pcount + ncount)*100 / total) + "%"
                print " objective".ljust(16,"-") + "> %.2f" % (100 - ((float)(pcount + ncount)*100 / total)) + "%"
                print Tcolors.C
                print Tcolors.BGGRAY
                polarity_scores = {'postive': 0, 'neutral': 0, 'negative': 0}
                for sense in results.keys():
                    count = results[sense]['count']
                    percentage = (float)(count) * 100 / (len(sentences))
                    polarity_scores[sense] = percentage
                    print " " +sense.ljust(15,"-")+"> %.2f" % (percentage) + "%"
                  
                print Tcolors.C 
                ssum = sum(scores)
                confidence = " (%.2f, %.2f)" % (ssum,sum(nscores))
                final_sent = ""
                pos = True
                if results["negative"]["count"] > len(sentences)*1.0/3:
                    pos = False

                # Print total sentiment score and normalized sentiment score
                if ssum > 0 and pos:
                    print Tcolors.RES + Tcolors.OKGREEN + " positive" + confidence + Tcolors.C
                    final_sent = "positive"
                elif ssum == 0:
                    print Tcolors.RES + Tcolors.OKGREEN +  " neutral" + confidence + Tcolors.C
                    final_sent = "neutral"
                else:
                    print Tcolors.RES + Tcolors.OKGREEN +  " negative" + confidence + Tcolors.C
                    final_sent = "negative"
                print Tcolors.C
                
                # Store results
                total_result_hash = {'sentences' : sentences,
                                     'sentiments': sentiments,
                                     'scores'    : scores,
                                     'nscores'   : nscores,
                                     'results'   : results,
                                     'subjective': subjective,
                                     'objective' : objective,
                                     'positive'  : polarity_scores['positive'],
                                     'neutral'   : polarity_scores['neutral'],
                                     'negative'  : polarity_scores['negative'],
                                      'final' : {final_sent:{'score':ssum,'nscore':sum(nscores)}}} 
                print "total_result_hash" + str(total_result_hash['objective']) + str(total_result_hash['positive']) + str(total_result_hash['negative']) + str(total_result_hash['neutral'])
        # Train SVM classifier
        # self.train_svm()
        return total_result_hash
    
    def normalize(self, text):
        """
            Make some word improvements before feeding to the sentence tokenizer.
        """  
        rr = RepeatReplacer(self.lexicon)
        normalized_text = []
        final = None
        try:
            for word in text.split():
                normal = rr.replace(word.lower()) 
                if word[0].isupper(): 
                    normal = normal[0].upper() + normal[1:]
                
                normalized_text.append(normal)
                final = " ".join(normalized_text)
        except:
                final = text
    
        return final
                
    def train_svm(self):
        """
            Train SVM and store data with pickle.
        """
        self.svm.train(self.total_sentences, self.total_sentiments)
        t_output = open(self.svm_train_filename,'wb')
        l_output = open(self.svm_label_filename,'wb')
        pickle.dump(self.total_sentences,t_output)
        pickle.dump(self.total_sentiments,l_output)
        t_output.close()
        l_output.close()

'''
if __name__ == '__main__':       
    sentiment = Sentiment()
    if len(sys.argv) > 1:
        sentiment.analyze([sys.argv[1]]) 
    else:
        sentiment.analyze([u"I was blown away by some of the comments here posted by people who is either uneducated, ignorant, self-righteous or all-of-the-above...I'm irritated and saddened as I read these \"finger-pointing\" or \"I'm right and you're wrong\" type of posts! Grow up folks! You're not in grade school...learn to embrace what is positive and move forward to do what is right... I have to give much love and respect to Ronny...your work is AMAZING!!! You cannot fathom how good I feel after I watched this video...regardless of history, politics, or whatever forces that makes what the mid-east today...for what you did and many of the followers in Iran and Palestine ...I BELIEVE TOMORROW WILL BE BETTER!!!!!! My name is Christopher Lee, I'm a nurse in Los Angeles and I {HEART} YOU ALL (especially to all of you beautiful and sweet ladies across the way)!!!!!"])      
'''
if __name__ == '__main__':       
    sentiment = Sentiment()
    sentence_polarity_infomap = {}
    filepath = 'stocknews/Combined_News_DJIA.csv'
    combined_stock_data = pd.read_csv(filepath)
    combined_stock_data['Para'] = combined_stock_data['Top1']
    for x in range(2, 26):
        combined_stock_data['Para'] += combined_stock_data['Top'+str(x)]

    for index, sentence in combined_stock_data['Para'].iteritems():
        sentence_polarity_infomap = sentiment.analyze([sentence])
        print "polarity" + str(sentence_polarity_infomap)
        if not (sentence_polarity_infomap == {}):
            combined_stock_data.at[index, 'Subjectivity'] = sentence_polarity_infomap['subjective']
            combined_stock_data.at[index, 'Objectivity'] = sentence_polarity_infomap['objective']
            combined_stock_data.at[index, 'Positive'] = sentence_polarity_infomap['positive']
            combined_stock_data.at[index, 'Neutral'] = sentence_polarity_infomap['neutral']
            combined_stock_data.at[index, 'Negative'] = sentence_polarity_infomap['negative']
        #combined_stock_data['Subjectivity'][index] = sentence_polarity_infomap['subjective']
        """
        combined_stock_data['Objectivity'][index] = sentence_polarity_infomap['objective']
        combined_stock_data['Positive'][index] = sentence_polarity_infomap['positive']
        combined_stock_data['Neutral'][index] = sentence_polarity_infomap['neutral']
        combined_stock_data['Negative'][index] = sentence_polarity_infomap['negative']
        """
    print combined_stock_data['Subjectivity'].shape
    print combined_stock_data['Objectivity'].shape
    print combined_stock_data['Positive'].shape
    print combined_stock_data['Neutral'].shape
    print combined_stock_data['Negative'].shape
    print combined_stock_data['Subjectivity'].head(20)
    print combined_stock_data['Objectivity'].head(20)
    print combined_stock_data['Positive'].head(20)
    print combined_stock_data['Neutral'].head(20)
    print combined_stock_data['Negative'].head(20)
    combined_stock_data.to_csv('combined_stock_data.csv')
