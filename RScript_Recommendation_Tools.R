########## Recommendation Tools - Last.fm ###########

# install packages
install.packages("reshape")
library(reshape2)
library(tidyr)
library(plyr)
library(data.table)
library(proxy)

###################################
##### Part 1 - Reading Data #######
###################################

user_taggedartists <- read.delim("user_taggedartists.dat", header = TRUE)
tags <- read.delim("tags.dat", header = TRUE)
user_artists <- read.delim("user_artists.dat", header = TRUE)

###### content-based

#merge tables
basematrix_usertags <- merge(user_taggedartists, tags, by.x = "tagID", by.y = "tagID")

# create the content based basetable
user_artists_dcast <- dcast(basematrix_usertags[, c(1,3,7)], artistID ~ tagValue, value.var="tagID", fun=sum)
user_artists_dcast[,-1] <- ifelse(user_artists_dcast[,-1] >=1, 1, 0)

# save and load the basetable
saveRDS(user_artists_dcast, file = "CB_basetable.Rds")
CB_basetable <- readRDS("CB_basetable.Rds")

######### collaborative filtering

hist(user_artists$weight[user_artists$weight<1500], breaks = 50)

# categorize the weight into 4 categories (1 refers to low, 4 refers to high)
user_artists$weight1 <- ifelse(user_artists$weight < 100, "1", (ifelse(user_artists$weight >= 100 & user_artists$weight <250, "2", (ifelse(user_artists$weight >= 250 & user_artists$weight <600, "3", "4")))))
table(user_artists$weight1)

# create the collaborative filtering basetable
user_artists_spread <- spread(user_artists[, c(1,2,4)], artistID, weight1)

# save and load the basetable
saveRDS(user_artists_spread, file = "CF_basetable.Rds")
CF_basetable <- readRDS("CF_basetable.Rds")
CF_basetable1 <- readRDS("CF_basetable.Rds")

# subset the basetable (too large to run the functions)

CB_basetable_sample <- sample(1:nrow(CB_basetable), 0.05*nrow(CB_basetable))

CB_basetable <- CB_basetable[CB_basetable_sample,]

# Matching ArtistID in CF_Basetable and CB_Basetable
CF_ArtistID_Unique<-unique(names(CF_basetable))
CB_ArtistID_Unique<-unique(CB_basetable[,1])
ArtistID_Intersect<-intersect(CF_ArtistID_Unique,CB_ArtistID_Unique)

CB_basetable <- CB_basetable[CB_basetable$artistID %in% ArtistID_Intersect,]

CF_basetable <- CF_basetable[,colnames(CF_basetable) %in% ArtistID_Intersect]

# CF_basetable as matrix
CF_matrix<- data.matrix(CF_basetable, rownames.force = NA)
userID <- CF_basetable1[,1]
rownames(CF_matrix) <- userID

# Reduce CB_basetable in order to apply the function faster
CB_basetable <- CB_basetable[, 1:100]

saveRDS(CB_basetable, file = "CB_basetable_sample.Rds")
saveRDS(CF_basetable, file = "CF_basetable_sample.Rds")

CB_basetable <- readRDS("CB_basetable_sample.Rds")
CF_basetable <- readRDS("CF_basetable_sample.Rds")

######################################
###### Part2 - Create Functions ######
######################################

##### UBCF as a function - User-based CF for all users #####

UserBasedCF <- function(train_data, test_data, N, NN, onlyNew=TRUE){
  
  ### similarity ###
  similarity_matrix <- matrix(, nrow = nrow(test_data), ncol = nrow(train_data), 
                              dimnames = list(rownames(test_data), rownames(train_data)))
  
  for (i in rownames(test_data)){
    for (j in rownames(train_data)){
      sim <- sum((test_data[i, ]-mean(test_data[i,]))*(train_data[j,]-mean(train_data[j,])), na.rm=TRUE)/sqrt(sum((test_data[i, ]-mean(test_data[i, ]))^2, na.rm=TRUE) * sum((train_data[j, ]-mean(train_data[j, ]))^2, na.rm=TRUE))
      similarity_matrix[i,j] <- sim
    }
  } 
  print("similarity calculation done")
  ### Nearest Neighbors ###
  similarity_matrix_NN <- similarity_matrix
  
  for (k in 1:nrow(similarity_matrix_NN)){
    crit_val <- -sort(-similarity_matrix_NN[k,])[NN]
    similarity_matrix_NN[k,] <- ifelse(similarity_matrix_NN[k,] >= crit_val, similarity_matrix_NN[k,], NA)
  }
  
  print("Nearest Neighbor selection done")
  ### Prediction ###
  # Prepare
  prediction <- matrix(, nrow=nrow(test_data), ncol(test_data), 
                       dimnames=list(rownames(test_data), colnames(test_data)))
  prediction2 <- matrix(, nrow=nrow(test_data), ncol(test_data), 
                        dimnames=list(rownames(test_data), colnames(test_data)))
  
  TopN <- matrix(, nrow=nrow(test_data), ncol=N, dimnames=list(rownames(test_data)))
  ### Numerator ###
  for (u in rownames(test_data)){
    similarity_vector <- na.omit(similarity_matrix_NN[u, ])
    
    NN_norm <- train_data[rownames(train_data) %in% names(similarity_vector),]
    
    CM <- colMeans(train_data, na.rm=TRUE)
    for (l in 1:ncol(NN_norm)){
      NN_norm[,l] <- NN_norm[,l] - CM[l]
    }
    NN_norm[is.na(NN_norm)] <- 0
    
    # Numerator
    Num = similarity_vector %*% NN_norm
    
    #Prediction
    prediction[u, ] =  mean(test_data[u, ], na.rm=TRUE)  + (Num/sum(similarity_vector, na.rm=TRUE))
    
    
    if (onlyNew == TRUE){
      unseen <- names(test_data[u, is.na(test_data[u,])])
      prediction2[u, ] <- ifelse(colnames(prediction) %in% unseen, prediction[u, ], NA)
    }else{
      prediction2[u, ] <- prediction[u, ]
    }
    
    TopN[u, ] <- names(-sort(-prediction2[u, ])[1:N])
    
  }
  
  print("Prediction done")
  
  res <- list(prediction, TopN)
  names(res) <- c('prediction', 'topN')
  
  return(res)
}

###### Item-based CF as a function ######

ItemBasedCF <- function(train_data, test_data, N, NN, onlyNew=TRUE){
  # Similarity
  
  similarity_matrix <- as.matrix(simil(t(train_data), method="pearson"))
  
  print("Similarity calculation done")
  # Nearest Neighbor
  similarity_matrix_NN <- similarity_matrix
  
  for (k in 1:ncol(similarity_matrix_NN)){
    crit_val <- -sort(-similarity_matrix_NN[,k])[NN]
    similarity_matrix_NN[,k] <- ifelse(similarity_matrix_NN[,k] >= crit_val, similarity_matrix_NN[,k], NA)
  }
  similarity_matrix_NN[is.na(similarity_matrix_NN)] <- 0
  
  train_data[is.na(train_data)] <- 0
  
  test_data2 <- test_data
  test_data2[is.na(test_data2)] <- 0
  
  print("Nearest neighbor selection done")
  
  ### Prediction ###
  prediction <- matrix(, nrow=nrow(test_data), ncol=ncol(test_data), 
                       dimnames=list(rownames(test_data), colnames(test_data)))
  prediction2 <- matrix(, nrow=nrow(test_data), ncol(test_data), 
                        dimnames=list(rownames(test_data), colnames(test_data)))
  TopN <- matrix(, nrow=nrow(test_data), N, dimnames=list(rownames(test_data)))
  
  for (u in rownames(test_data)){
    # Numerator
    Num <-  test_data2[u, ] %*% similarity_matrix_NN
    
    # Denominator
    Denom <- colSums(similarity_matrix_NN, na.rm=TRUE)
    
    # Prediction
    prediction[u, ] <- Num/Denom
    
    if (onlyNew == TRUE){
      unseen <- names(test_data[u, is.na(test_data[u,])])
      prediction2[u, ] <- ifelse(colnames(prediction) %in% unseen, prediction[u, ], NA)
    }else{
      prediction2[u, ] <- prediction[u, ]
    }
    
    TopN[u, ] <- names(-sort(-prediction2[u, ])[1:N])
    
  }
  
  print("Prediction done")
  
  res <- list(prediction, TopN)
  names(res) <- c('prediction', 'topN')
  
  return(res)
}

###### Cluster based CF as a function ######

ClusterBasedCF <- function(data, N, centers, iter, onlyNew=TRUE){
  
  data2 <- data
  
  # fill with average product rating
  colmeans <- colMeans(data2, na.rm=TRUE)
  
  for (j in colnames(data2)){
    data2[, j] <- ifelse(is.na(data2[ ,j]), colmeans[j], data2[, j])
  }
  
  km <- kmeans(data2, centers=centers, iter.max=iter)
  
  head(km$cluster)
  head(km$centers)
  
  
  # Statistics of the groups
  tab <- table(km$cluster)
  
  # Assign users to groups
  RES <- cbind(data, as.data.frame(km$cluster))
  
  # Calculate average ratings for everi cluster
  aggregation <- aggregate(RES, list(RES$"km$cluster"), mean, na.rm=T)
  aggregation <- aggregation[,-1]
  
  # Make a prediction
  users <- as.data.frame(RES$"km$cluster")
  users <- cbind(users, rownames(RES))
  colnames(users) <- c("km$cluster", 'rn')
  
  
  prediction = merge(users, aggregation, by="km$cluster")
  rownames(prediction) <- prediction$rn
  
  prediction  <- prediction[order(rownames(prediction)), -1:-2]
  
  prediction2 <- matrix(, nrow=nrow(prediction), ncol(prediction), 
                        dimnames=list(rownames(prediction), colnames(prediction)))
  colnames(prediction2) <- colnames(prediction)
  rownames(prediction2) <- rownames(prediction)
  
  for (u in rownames(prediction)){
    if (onlyNew == TRUE){
      unseen <- names(data[u, is.na(data[u,])])
      
      prediction2[u, ] <- as.numeric(t(ifelse(colnames(prediction) %in% unseen, prediction[u, ], as.numeric(NA))))
    }else{
      prediction2[u, ] <- prediction[u, ]
    }
  }
  
  # TopN
  TopN <- t(apply(prediction, 1, function(x) names(head(sort(x, decreasing=TRUE), 5))))
  
  print("Prediction done")
  
  res <- list(prediction, TopN)
  names(res) <- c('prediction', 'topN')
  
  return(res)
}

####### Content Based Function #####

ContentBased <- function(product_data, test_data, N, NN, onlyNew=TRUE){
  
  # Similarity calculation (stolen from user-based CF)
  similarity_matrix <- as.matrix(simil(product_data, method="pearson"))
  
  print("Similarity calculation done")
  
  # Set Nearest neighbors (stolen from user-based CF)
  similarity_matrix_NN <- similarity_matrix
  
  for (k in 1:nrow(similarity_matrix_NN)){
    crit_val <- -sort(-similarity_matrix_NN[k,])[NN]
    similarity_matrix_NN[k,] <- ifelse(similarity_matrix_NN[k,] >= crit_val, similarity_matrix_NN[k,], 0)
  }
  
  similarity_matrix_NN[is.na(similarity_matrix_NN)] <- 0
  test_data2 <- test_data
  test_data2[is.na(test_data2)] <- 0
  
  print("Nearest neighbor selection done")
  
  ### Prediction (stolen from item-based CF) ###
  prediction <- matrix(, nrow=nrow(test_data), ncol=ncol(test_data), 
                       dimnames=list(rownames(test_data), colnames(test_data)))
  prediction2 <- matrix(, nrow=nrow(test_data), ncol(test_data), 
                        dimnames=list(rownames(test_data), colnames(test_data)))
  TopN <- matrix(, nrow=nrow(test_data), N, dimnames=list(rownames(test_data)))
  
  for (u in rownames(test_data)){
    # Numerator
    Num <-  test_data2[u, ] %*% similarity_matrix_NN
    
    # Denominator
    Denom <- colSums(similarity_matrix_NN, na.rm=TRUE)
    
    # Prediction
    prediction[u, ] <- Num/Denom
    
    if (onlyNew == TRUE){
      unseen <- names(test_data[u, is.na(test_data[u,])])
      prediction2[u, ] <- ifelse(colnames(prediction) %in% unseen, prediction[u, ], NA)
    }else{
      prediction2[u, ] <- prediction[u, ]
    }
    
    TopN[u, ] <- names(-sort(-prediction2[u, ])[1:N])
    
  }
  
  print("Prediction done")
  
  res <- list(prediction, TopN)
  names(res) <- c('prediction', 'topN')
  
  return(res)
}


######### MAE Function ########

MAE <- function(prediction, real){
  
  if (nrow(prediction) == nrow(real) & ncol(prediction) == ncol(real)){
    MAE = sum( abs((prediction - real)^2) , na.rm = TRUE ) / (nrow(prediction) * ncol(prediction))
    return(MAE)
  }else{
    return("Dimension of prediction are not equal to dimension of real")
  }
}

######## F1 Function #########

F1 <- function(prediction, real, threshold=NA, TopN=NA){
  if (nrow(prediction) == nrow(real) & ncol(prediction) == ncol(real)){
    # Threshold #
    if (!is.na(threshold)){
      TP = sum(ifelse(prediction >= threshold & real >= threshold, 1, 0), na.rm=T)
      FP = sum(ifelse(prediction >= threshold & real < threshold, 1, 0), na.rm=T)
      FN = sum(ifelse(prediction < threshold & real >= threshold, 1, 0), na.rm=T)
      Recall = TP/(TP+FN)
      Precision = TP/(TP+FP)
      F1 <- 2*((Precision*Recall)/(Precision + Recall))
      Class_Thres = list(Recall, Precision, F1)
      names(Class_Thres) = c("Recall", "Precision", "F1")
    }
    if (!is.na(TopN)){
      TP = vector(, length = nrow(prediction))
      FP = vector(, length = nrow(prediction))
      FN = vector(, length = nrow(prediction))
      
      for (i in nrow(prediction)){
        threshold_pred = -sort(-prediction[i, ])[TopN]
        threshold_real = -sort(-real[i, ])[TopN]
        TP[i] = sum(ifelse(prediction[i, ] >= threshold_pred & real[i, ] >= threshold_real, 1, 0), na.rm=T)
        FP[i] = sum(ifelse(prediction[i, ] >= threshold_pred & real[i, ] < threshold_real, 1, 0), na.rm=T)
        FN[i] = sum(ifelse(prediction[i, ] < threshold_pred & real[i, ] >= threshold_real, 1, 0), na.rm=T)
      }
      TP = sum(TP[i])
      FP = sum(FP[i])
      FN = sum(FN[i])
      Recall = TP/(TP+FN)
      Precision = TP/(TP+FP)
      F1 <- 2*((Precision*Recall)/(Precision + Recall))
      Class_TopN = list(Recall, Precision, F1)
      names(Class_TopN) = c("Recall", "Precision", "F1")
    }
    
    
    if (!is.na(threshold) & !is.na(TopN)){
      Class = list(Class_Thres, Class_TopN)
      names(Class) = c("Threshold", "TopN")
    }else if (!is.na(threshold) & is.na(TopN)) {
      Class = Class_Thres
    }else if (is.na(threshold) & !is.na(TopN)) {
      Class = Class_TopN
    }else{
      Class = "You have to specify the 'Threshold' or 'TopN' parameter!"
    }
    return(Class)
    
  }else{
    return("Dimension of prediction are not equal to dimension of real")
  }
}

RSME <- function(prediction, real){
  
  if (nrow(prediction) == nrow(real) & ncol(prediction) == ncol(real)){
    RSME = sqrt( sum( (prediction - real)^2 , na.rm = TRUE ) / (nrow(prediction) * ncol(prediction)) )
    return(RSME)
  }else{
    return("Dimension of prediction are not equal to dimension of real")
  }
}

####################################################################
##### Part 3 - Computing and Evaluating Recommendation Systems #####
####################################################################

####### a) Compute CF RecSys #######

##### Split train - Test

set.seed(2)
train_rows <- sample(1:nrow(CF_matrix), 0.7*nrow(CF_matrix))

train <- CF_matrix[train_rows,]
test <- CF_matrix[-train_rows,]

##### Compute User Based RecSys

UBCF <- UserBasedCF(train, test, N=5, NN=10, onlyNew=TRUE)

predictionUBCF <- as.data.frame(UBCF$prediction)
topNUBCF <- as.data.frame(UBCF$topN)

##### Compute Item Based RecSys

IBCF <- ItemBasedCF(train, test, N=3, NN=10, onlyNew=TRUE)

predictionIBCF <- as.data.frame(IBCF$prediction)
topNIBCF <- as.data.frame(IBCF$topN)

####### b) Compute CB RecSys #######

CB <- ContentBased(CB_basetable, CF_matrix, 3, 10, onlyNew=T)

predCB <- CB$prediction
topNCB <- CB$topN

######## c) Compare and Evaluate Models #######

### Prediction Accuracy ###

#### Content-based
# RSME
RSME(CB$prediction, test)

#MAE

MAE(CB$prediction, test)

#### User-based
# RSME
RSME(UBCF$prediction, test)

#MAE
MAE(UBCF$prediction, test)

#### Item-based
# RSME
RSME(IBCF$prediction, test)

# MAE
MAE(IBCF$prediction, test)

### Classification Accuracy ###

#### F1 Content-based
F1(CB$prediction, test, threshold=3)

#### F1 User-based
# F1
F1(UBCF$prediction, test, threshold=3)

#### F1 Item-based
F1(IBCF$prediction, test, threshold=3)

######################
#### Hybrid systems###
######################

### Split train - Test ###
set.seed(2)
train_rows = sample(1:nrow(CF_basetable), 0.7*nrow(CF_basetable))

train <- (CF_matrix)[train_rows,]
test <- (CF_matrix)[-train_rows,]
test1 <- test[270, ]
test2 <- test[298, ]

### Compute individual models ###
CBTFIDF <- CB
IB <- IBCF

### Transform results to lists (to be able to use the rowMeans function) ###
CBTFIDF_list <- as.list(CBTFIDF$prediction)
IB_list <- as.list(IB$prediction)

####################
### Compute Mean ###
####################
hybrid <- rowMeans(cbind(as.numeric(CBTFIDF_list), as.numeric(IB_list)), na.rm=T)

### Transform list back to matrix with correct number of dimensions ###
Hybrid_prediction <- matrix(hybrid, nrow=nrow(test1), ncol=ncol(test1))
rownames(Hybrid_prediction) <- rownames(test1)
colnames(Hybrid_prediction) <- colnames(test1)

### Evaluate ###
# RSME and MAE
RSME(CBTFIDF$prediction, test1)
RSME(IB$prediction, test1)
RSME(Hybrid_prediction, test1)

MAE(CBTFIDF$prediction, test1)
MAE(IB$prediction, test1)
MAE(Hybrid_prediction, test1)

# Classification
F1(CBTFIDF$prediction, test1, threshold=5)
F1(IB$prediction, test1, threshold=5)
F1(Hybrid_prediction, test1, threshold=5)

#########################
### Linear Regression ###
#########################

# Train a linear Regression
### flatten test1 dataset
test_list <- as.list(test1)

### Transform list and matrices to dataframe
test_df <- data.frame(matrix(unlist(test_list), byrow=T))
CBTFIDF_df <- data.frame(matrix(unlist(CBTFIDF_list), byrow=T))
IB_df <- data.frame(matrix(unlist(IB_list), byrow=T))

### Combine created dataframes
input <- cbind(test_df, CBTFIDF_df, IB_df)
colnames(input) <- c('TARGET', 'CBTFIDF', 'IB')

### Train the linear regression
fit <- lm(TARGET ~ CBTFIDF + IB, data=input)
summary(fit)

### Score Models
CBTFIDF2 <- ContentBased(tfidf, test2, 3, 10, onlyNew=F)
IB2 <- ItemBasedCF(train, test2, 3, 10, onlyNew=F)

### Matrix to list
test_list2 <- as.list(test2)
CBTFIDF_list2 <- as.list(CBTFIDF2$prediction)
IB_list2 <- as.list(IB2$prediction)

### List to dataframe
test_df2 <- data.frame(matrix(unlist(test_list2), byrow=T))
CBTFIDF_df2 <- data.frame(matrix(unlist(CBTFIDF_list2), byrow=T))
IB_df2 <- data.frame(matrix(unlist(IB_list2), byrow=T))

### combine dataframes to have an input dataset for linear regression
input2 <- cbind(test_df2, CBTFIDF_df2, IB_df2)
colnames(input2) <- c('TARGET', 'CBTFIDF', 'IB')

### Predict using the model calculated on test2 dataset
Hybrid_lin_reg <- predict(fit, input2)

### Transform the list of results to matrix with the correct dimensions
Hybrid_lin_reg <- matrix(Hybrid_lin_reg, nrow=nrow(test2), ncol=ncol(test2))
rownames(Hybrid_lin_reg) <- rownames(test2)
colnames(Hybrid_lin_reg) <- colnames(test2)

# Evaluation
# RSME
RSME(CBTFIDF2$prediction, test2)
RSME(IB2$prediction, test2)
RSME(Hybrid_lin_reg, test2)

MAE(CBTFIDF2$prediction, test2)
MAE(IB2$prediction, test2)
MAE(Hybrid_lin_reg, test2)

# Classification
F1(CBTFIDF2$prediction, test2, threshold=5)
F1(IB2$prediction, test2, threshold=5)
F1(Hybrid_lin_reg, test2, threshold=5)

################################################################################
