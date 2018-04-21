################################################################################################################
#Project Description - This program performs sentiment analysis from tweets while comparing two different topics
#Author - Nisha Muthukumaran
#Date Created - 18th April, 2018
#Contact info - nm8111@wildcats.unh.edu
#Modifications - Version 1: Tweets for 18th April
#                Version 2: Tweets for 21st April
################################################################################################################


#install the below libraries
setwd("C:/Users/nisha/Documents/DATA 902/NLP/Project/")
library(qdap)
library(readtext)
library(rvest)
library(tm)
library(wordcloud)
library(RWeka)
library(tidytext)
library(dplyr)
library(tidyr)
library(radarchart)
library(ggplot2)

#reading the texts files generated from Python into R
syria <- readtext("syria.txt")
avengers <- readtext("avengers.txt")

#contains a combination of both files
speech <- c(syria$text,avengers$text)

#making a corpus of a vector source 
speech_corpus <- VCorpus(VectorSource(speech))

# pre_processing function
clean_corpus <- function(corpus){
  
  #this variable is a function that removes links that begin with https 
  removeURL <- content_transformer(function(x) gsub("(f|ht)tp(s?)://\\S+", "", x, perl=T))
  
  cleaned_corpus <- tm_map(corpus, removeURL)
  
  #this converts all the words in the tweets to lower case alphabets
  cleaned_corpus <- tm_map(cleaned_corpus, content_transformer(tolower))
  
  #this gets rid of all the punctuations like !,?.
  cleaned_corpus <- tm_map(cleaned_corpus, removePunctuation)
  
  #this gets rid of numeric digits in the tweets
  cleaned_corpus <- tm_map(cleaned_corpus, removeNumbers)
  
  #this gets rid of all the stopwords in the tweets like articles, pronouns
  cleaned_corpus <- tm_map(cleaned_corpus, removeWords, stopwords("english"))
  
  #this creates a list of custom stop words and gets rid of custom words which the user can input and is topic related
  custom_stop_words <- c("syria","amp","avengers","syrian","syrianwar","infinitywar", "avengersinfinitywar")
  
  cleaned_corpus <- tm_map(cleaned_corpus, removeWords, custom_stop_words)
  
  #this gets rid of the white spaces in the doc
  cleaned_corpus <- tm_map(cleaned_corpus, stripWhitespace)
  
  return(cleaned_corpus) #returns the cleaned doc
}


#applying the cleaning function onto the corpus
cleaned_speech_corpus <- clean_corpus(speech_corpus)

################################################converts the cleaned corpus to TDM/DTM##################################################
TDM_speech <- TermDocumentMatrix(cleaned_speech_corpus)

TDM_speech_m <- as.matrix(TDM_speech)

# Term Frequency of all the words on how often they were found
term_frequency <- rowSums(TDM_speech_m)

# Sort term_frequency in descending order
term_frequency <- sort(term_frequency,dec=TRUE)

# View the top 20 most common words
top10 <- term_frequency[1:20]

# Plot a barchart of the 20 most common words
barplot(top10,col="turquoise",las=2)


###############################Commonality Cloud - commonality function gives similarities in the tweets###############################

commonality.cloud(TDM_speech_m,colors=brewer.pal(8, "Paired"),random.order=FALSE,max.words = 300)

##################################Comparison Cloud - what was unique to each topic(movie/war) in the tweets############################

TDM_speech <- TermDocumentMatrix(cleaned_speech_corpus)

colnames(TDM_speech) <- c("Syria","Avengers")

TDM_speech_m <- as.matrix(TDM_speech)

comparison.cloud(TDM_speech_m,colors=brewer.pal(8, "Set1"),max.words = 200)

#########################################################Unigram cloud################################################################

#creating a function that allows only 1 word to be accounted for at a time
tokenizer <- function(x)
  NGramTokenizer(x,Weka_control(min=1,max=1))

unigram_tdm <- TermDocumentMatrix(cleaned_speech_corpus,control = list(tokenize=tokenizer)) #applying the unigram function to corpus

unigram_tdm_m <- as.matrix(unigram_tdm)#converting it to a matrix

# Term Frequency
term_frequency <- rowSums(unigram_tdm_m)

# Sort term_frequency in descending order
term_frequency <- sort(term_frequency,dec=TRUE)

# Create word_freqs
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)

# Create a wordcloud for the values in word_freqs
wordcloud(word_freqs$term, word_freqs$num,min.freq=10,max.words=200,colors=brewer.pal(8, "Paired"))


###################################################Bigrams clouds#########################################################

#creating a function that allows 2 adjascent words to be accounted for at a time
tokenizer <- function(x)
  NGramTokenizer(x,Weka_control(min=2,max=2))

bigram_tdm <- TermDocumentMatrix(cleaned_speech_corpus,control = list(tokenize=tokenizer))#applying the bigram function to corpus

bigram_tdm_m <- as.matrix(bigram_tdm)#converting it to a matrix

# Term Frequency
term_frequency <- rowSums(bigram_tdm_m)

# Sort term_frequency in descending order
term_frequency <- sort(term_frequency,dec=TRUE)

# Create word_freqs
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)

# Create a wordcloud for the values in word_freqs 
wordcloud(word_freqs$term, word_freqs$num,min.freq=2,max.words=200,colors=brewer.pal(8, "Paired"))

###################################################Trigram Clouds#########################################################

#creating a function that allows 3 adjascent words to be accounted for at a time
tokenizer <- function(x)
  NGramTokenizer(x,Weka_control(min=3,max=3))

trigram_tdm <- TermDocumentMatrix(cleaned_speech_corpus,control = list(tokenize=tokenizer))#applying the trigram function to corpus

trigram_tdm_m <- as.matrix(trigram_tdm)#converting it to a matrix

#Term Frequency
term_frequency <- rowSums(trigram_tdm_m)

# Sort term_frequency in descending order
term_frequency <- sort(term_frequency,dec=TRUE)

# Create word_freqs
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)

# Create a wordcloud for the values in word_freqs
wordcloud(word_freqs$term, word_freqs$num,min.freq=8,max.words=300,colors=brewer.pal(8, "Paired"))

#################################################Tf-idf weighting########################################################

tfidf_tdm <- TermDocumentMatrix(cleaned_speech_corpus,control=list(weighting=weightTfIdf))#applying tf-idf weighting as a parameter

tfidf_tdm_m <- as.matrix(tfidf_tdm)

# Term Frequency
term_frequency <- rowSums(tfidf_tdm_m)

# Sort term_frequency in descending order
term_frequency <- sort(term_frequency,dec=TRUE)

# Create word_freqs
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)

# Create a wordcloud for the values in word_freqs
wordcloud(word_freqs$term, word_freqs$num,min.freq=5,random.oreder=FALSE,max.words=200,colors=brewer.pal(8, "Paired"))

################################################Lexicons using Tidy Text for NRC Radar chart#################################################

TDM_mytext <- TermDocumentMatrix(cleaned_speech_corpus)

#applying tidy function converts it to a dataframe
mytext_tidy <- tidy(TDM_mytext)


#NRC lexicon for Radar chart
nrc_lex <- get_sentiments("nrc") #gets the inbuilt NRC lexicon

story_nrc <- inner_join(mytext_tidy, nrc_lex, by = c("term" = "word")) #inner joins tidy dataframe to NRC lexicon

story_nrc_noposneg <- story_nrc[!(story_nrc$sentiment %in% c("positive","negative")),]

aggdata <- aggregate(story_nrc_noposneg$count, list(index = story_nrc_noposneg$sentiment), sum)

chartJSRadar(aggdata)


##############################################Lexicon for sentiment graphs###############################################
######SENTIMENT ANALYSIS FOR SYRIAN WAR######

mytext <- read.csv(file ="C:/Users/nisha/Documents/DATA 902/NLP/Project/syria.csv")

mytext$text <- iconv(mytext$Tweet, from = "UTF-8", to = "ASCII", sub = "")

mytext_corpus <- VCorpus(VectorSource(mytext$text))

cleaned_mytext_corpus <- clean_corpus(mytext_corpus)

TDM_mytext <- TermDocumentMatrix(cleaned_mytext_corpus)

mytext_tidy <- tidy(TDM_mytext)

# bing
bing_lex <- get_sentiments("bing")

mytext_bing_lex <- inner_join(mytext_tidy, bing_lex, by = c("term" = "word"))

mytext_bing_lex$sentiment_n <- ifelse(mytext_bing_lex$sentiment=="negative", -1, 1)

mytext_bing_lex$sentiment_value <- mytext_bing_lex$sentiment_n * mytext_bing_lex$count

mytext_bing_lex$document <- as.numeric(rownames(mytext_bing_lex)) ##

bing_aggdata1 <- aggregate(mytext_bing_lex$sentiment_value, list(index = mytext_bing_lex$document),sum)

sapply(bing_aggdata1,typeof)

bing_aggdata1$index <- as.numeric(bing_aggdata1$index)

colnames(bing_aggdata1) <- c("index","bing_score")


# afinn
afinn_lex <- get_sentiments("afinn")

mytext_afinn_lex <- inner_join(mytext_tidy, afinn_lex, by = c("term" = "word"))

mytext_afinn_lex$sentiment_value <- mytext_afinn_lex$score * mytext_afinn_lex$count

mytext_afinn_lex$document <- as.numeric(rownames(mytext_afinn_lex))

afinn_aggdata1 <- aggregate(mytext_afinn_lex$sentiment_value, list(index = mytext_afinn_lex$document), sum)

afinn_aggdata1$index <- as.numeric(afinn_aggdata1$index)

colnames(afinn_aggdata1) <- c("index","afinn_score")


# NRC
nrc_lex <- get_sentiments("nrc")

nrc_lex_pos_neg <- nrc_lex[nrc_lex$sentiment %in% c("positive","negative"),] 

mytext_nrc_lex <- inner_join(mytext_tidy, nrc_lex_pos_neg, by = c("term" = "word"))

mytext_nrc_lex$sentiment_n <- ifelse(mytext_nrc_lex$sentiment=="negative", -1, 1)

mytext_nrc_lex$sentiment_value <- mytext_nrc_lex$sentiment_n * mytext_nrc_lex$count

mytext_nrc_lex$document <- as.numeric(rownames(mytext_nrc_lex))

nrc_aggdata1 <- aggregate(mytext_nrc_lex$sentiment_value, list(index = mytext_nrc_lex$document), sum)

nrc_aggdata1$index <- as.numeric(nrc_aggdata1$index)

colnames(nrc_aggdata1) <- c("index","nrc_score")


#Merging all the above three lexicons into one plot

MyMerge <- function(x, y){
  df <- merge(x, y, by= "index", all.x= TRUE, all.y= TRUE)
  return(df)
}
all_sentiment_data1 <- Reduce(MyMerge, list(nrc_aggdata1, bing_aggdata1, afinn_aggdata1))

tidy_sentiment_data1 <- gather(all_sentiment_data1, sentiment_dict, sentiment_score, -index)

tidy_sentiment_data1[is.na(tidy_sentiment_data1)] <- 0

ggplot(data = tidy_sentiment_data1,aes(x=index,y=sentiment_score,fill=sentiment_dict))+
  geom_bar(stat="identity") + facet_grid(sentiment_dict~.)+theme_bw() + theme(legend.position = "none")+ggtitle("Syria twitter Sentiments")


###### SENTIMENT ANALYSIS FOR AVENGERS WAR######

mytweet <- read.csv(file ="C:/Users/nisha/Documents/DATA 902/NLP/Project/avengers.csv")

mytweet$text <- iconv(mytweet$Tweet, from = "UTF-8", to = "ASCII", sub = "")

mytweet_corpus <- VCorpus(VectorSource(mytweet$text))

cleaned_mytweet_corpus <- clean_corpus(mytweet_corpus)

TDM_mytweet <- TermDocumentMatrix(cleaned_mytweet_corpus)

mytweet_tidy <- tidy(TDM_mytweet)

# bing
bing_lex <- get_sentiments("bing")

mytweet_bing_lex <- inner_join(mytweet_tidy, bing_lex, by = c("term" = "word"))

mytweet_bing_lex$sentiment_n <- ifelse(mytweet_bing_lex$sentiment=="negative", -1, 1)

mytweet_bing_lex$sentiment_value <- mytweet_bing_lex$sentiment_n * mytweet_bing_lex$count

mytweet_bing_lex$document <- as.numeric(rownames(mytweet_bing_lex)) ##

bing_aggdata <- aggregate(mytweet_bing_lex$sentiment_value, list(index = mytweet_bing_lex$document),sum)

sapply(bing_aggdata,typeof)

bing_aggdata$index <- as.numeric(bing_aggdata$index)

colnames(bing_aggdata) <- c("index","bing_score")


# afinn
afinn_lex <- get_sentiments("afinn")

mytweet_afinn_lex <- inner_join(mytweet_tidy, afinn_lex, by = c("term" = "word"))

mytweet_afinn_lex$sentiment_value <- mytweet_afinn_lex$score * mytweet_afinn_lex$count

mytweet_afinn_lex$document <- as.numeric(rownames(mytweet_afinn_lex))

afinn_aggdata <- aggregate(mytweet_afinn_lex$sentiment_value, list(index = mytweet_afinn_lex$document), sum)

afinn_aggdata$index <- as.numeric(afinn_aggdata$index)

colnames(afinn_aggdata) <- c("index","afinn_score")


# NRC
nrc_lex <- get_sentiments("nrc")

nrc_lex_pos_neg <- nrc_lex[nrc_lex$sentiment %in% c("positive","negative"),] 

mytweet_nrc_lex <- inner_join(mytweet_tidy, nrc_lex_pos_neg, by = c("term" = "word"))

mytweet_nrc_lex$sentiment_n <- ifelse(mytweet_nrc_lex$sentiment=="negative", -1, 1)

mytweet_nrc_lex$sentiment_value <- mytweet_nrc_lex$sentiment_n * mytweet_nrc_lex$count

mytweet_nrc_lex$document <- as.numeric(rownames(mytweet_nrc_lex))

nrc_aggdata <- aggregate(mytweet_nrc_lex$sentiment_value, list(index = mytweet_nrc_lex$document), sum)

nrc_aggdata$index <- as.numeric(nrc_aggdata$index)

colnames(nrc_aggdata) <- c("index","nrc_score")


#Merging all the above three lexicons into one plot

MyMerge <- function(x, y){
  df <- merge(x, y, by= "index", all.x= TRUE, all.y= TRUE)
  return(df)
}
all_sentiment_data <- Reduce(MyMerge, list(nrc_aggdata, bing_aggdata, afinn_aggdata))

tidy_sentiment_data <- gather(all_sentiment_data, sentiment_dict, sentiment_score, -index)

tidy_sentiment_data[is.na(tidy_sentiment_data)] <- 0

ggplot(data = tidy_sentiment_data,aes(x=index,y=sentiment_score,fill=sentiment_dict))+
  geom_bar(stat="identity") + facet_grid(sentiment_dict~.)+theme_bw() + theme(legend.position = "none")+ggtitle("Avengers twitter Sentiments")




