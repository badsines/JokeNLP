# SET PATH

library(jsonlite)
library(dplyr)
library(tm)
library(SnowballC)
library(stringr)
library(slam)
library(tidytext)
library(tidyr)
library(topicmodels)
set.seed(123)


df <- fromJSON("reddit_jokes.json")


df$num_id <- strtoi(df$id, base = 36) #the id's are actually base 36


###################################################### - Remove duplicates based on body (not ideal)
dups<- df %>%
  select(body)%>%
  duplicated()
dups <- df[dups, c("body", "num_id", "score")]
nrow(dups) # number of exact duplicates

uniq_dups <- unique(dups$body)

uniq.df <- data_frame(body = unique(df$body), unq_id = as.factor(1:length(unique(df$body))))
df <- df %>%
  left_join(uniq.df) %>%
  group_by(unq_id) %>%
  slice(which.max(score))

df$document <- as.character(1:nrow(df))
###################################################### - Preprocessing and creating corpus

df$fullbody <- paste(df$title, df$body)                   # combine title and body
df$fullbody <- gsub("\r?\n|\r", " ", df$fullbody)         # remove new lines
df$fullbody <- gsub("\\.", " ", df$fullbody)              # remove periods
df$fullbody <- gsub("-", " ", df$fullbody, fixed = TRUE)  # remove "-"
df$fullbody <- gsub("[^[:alnum:]///' ]", "", df$fullbody) # remove anything that isn't alphanumeric

doc.vec <- VectorSource(df$fullbody)
doc.corpus <- Corpus(doc.vec)

doc.corpus <- tm_map(doc.corpus, content_transformer(tolower))
doc.corpus <- tm_map(doc.corpus, removeNumbers)
doc.corpus <- tm_map(doc.corpus, removePunctuation)
doc.corpus <- tm_map(doc.corpus, removeWords, stopwords('english'))
doc.corpus <- tm_map(doc.corpus, stripWhitespace)


###################################################### - Creating and pre-processing document-term matrix
dtm <- DocumentTermMatrix(doc.corpus)

# Remove words that occur less than 5 times
a<-col_sums(dtm, na.rm = TRUE)                            
sum(as.numeric(a<5))
dtm <- dtm[,!(a<5)]

# Remove documents that now have no words (aka, only had words that occurred less than 5 times)
a<-row_sums(dtm, na.rm = TRUE)==0
dtm <- dtm[!a,] # 28k words

###################################################### - Fitting LDA model

burnin <- 1500 
iter <- 1500
keep <- 100 

fitted <- LDA(dtm, k = 50, method = "Gibbs",
              control = list(burnin = burnin, iter = iter, keep = keep))


ldaTopics <- topics(fitted) # joke topic assignments
topicProbs <- as.data.frame(fitted@gamma)

###################################################### - Get corresponding dataframe
# Since we removed some rows in the dtm we need to do 
# the same to the original dataset

remDocs <- as.character(which(a))
df_f <- df[!(df$document %in% remDocs),] %>% ungroup()
df_f$topic <- ldaTopics
