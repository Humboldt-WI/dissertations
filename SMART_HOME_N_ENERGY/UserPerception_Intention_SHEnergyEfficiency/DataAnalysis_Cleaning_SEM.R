## HUMBOLDT-UNIVERSITAET ZU BERLIN   
## CHAIR OF INFORMATION SYSTEMS   

# Name:             Hee-Eun Lee
## Master's thesis: Understanding User Perception and Intention to Use Smart Home for Energy Efficiency using Structural Equation Modeling and Machine Learning###
# Due date:         15.01.2022
# 1st Examiner:     Prof. Dr. Lessmann
# 2nd Examiner:     PD Dr. Fabian
# Supervisor:       Dr. Alona Zharova

#------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------  INPUT NEEDED - SET WORKING DIRECTORY  ------------------------------------------------

# Please run RStudio as administrator for permissions
setwd("C:\\Users\\heeeun\\Dropbox\\Uni\\Masterarbeit\\Umfragedaten\\20211231")
getwd()
#------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------  DATA IMPORT & LIBRARIES  --------------------------------------------------------

# Install packages + libraries
install.packages("pacman")
library(pacman)
pacman::p_load(readxl, psych, plyr, semPlot, OpenMx, dplyr, ggplot2, data.table, stringr, knitr, kableExtra, scales, formattable,
               readr, forcats, gmodels, wordcloud, RColorBrewer, tidyverse, tm, NLP, devtools, 
               tidySEM, magrittr, MVN, car, corrplot, lattice, coefplot, wesanderson, ggridges, viridis, hrbrthemes,
               apaTables, Mplus, EQS, pander, semTable, plot.matrix, mice, GPArotation, QuantPsyc, pequod, rockchalk, 
               gvlma, magick, vtable, gt, mvnormtest, rstatix, seminr, rsvg)

# Import survey data and syntax
data <- as.data.frame(read_excel("surveydata.xlsx"))
source("surveysyntax.R")

#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------  DATA CLEANING  -------------------------------------------------------------

# omit unnecessary columns
df <- subset(data, select=-c(3,5:8,124:159))
#str(df) 


# omit incomplete entries
cleandf <- df[!is.na(df$submitdate), ]  #entries without submit date
cleandf <- cleandf[!cleandf$G01Q01=="No", ]  #participants that answered "No" to G01Q01=Are you currently living in a home that is equipped with a smart home system?
cleandf <- cleandf[!with(cleandf,is.na(cleandf$G01Q03) & is.na(cleandf$G01Q03_other)), ] #entries without naming smart home system

# omitting NA entries from question group 2
data2 <- cleandf[,c(58:64)]
cleandf <- cleandf[complete.cases(data2),] 

# omitting NA entries from question group 3
data3 <- cleandf[,c(65:89)]
cleandf <- cleandf[complete.cases(data3),] 

# omitting NA entries from question group 4
data4 <- cleandf[,c(90:103)]
cleandf <- cleandf[complete.cases(data4),] 

# omitting NA entries from question group 5 - demographics
data5 <- cleandf[,c(104:105, 110:111, 114:116)]
cleandf <- cleandf[complete.cases(data5),] 

# checking for NA in G05Q23, G05Q24, G05Q27, G01Q05, G01Q06, G01Q07
droprows <- c()
for (i in 1:nrow(cleandf)){
  row <- cleandf[i,]
  if(sum(is.na(cleandf[i,106:107]))>1){ #G05Q23
    droprows <- append(droprows,i)
    print(paste("Please drop Id:", cleandf[i,1], " in row", i))
  } else if(sum(is.na(cleandf[i,108:109]))>1){ #G05Q24
    droprows <- append(droprows, i)
    print(paste("Please drop Id:", cleandf[i,1], " in row", i))
  } else if(sum(is.na(cleandf[i,112:113]))>1){ #G05Q27
    droprows <- append(droprows, i)
    print(paste("Please drop Id:", cleandf[i,1], " in row", i)) 
  } else if(sum(is.na(cleandf[i,9:44]))>35){ #G01Q05
    droprows <- append(droprows, i)
    print(paste("Please drop Id:", cleandf[i,1], " in row", i))
  } else if(sum(is.na(cleandf[i,45:46]))>1){ #G01Q06
    droprows <- append(droprows, i)
    print(paste("Please drop Id:", cleandf[i,1], " in row", i))
  } else if(sum(is.na(cleandf[i, 47:57]))>10){ #G01Q07
    droprows <- append(droprows, i)
    print(paste("Please drop Id:", cleandf[i,1], " in row", i))
  }
}

cleandf <- cleandf[-c(unique(droprows)),] 
print(nrow(cleandf)) 

# create list for incomplete entries
incomplete <- data[-c(cleandf$id),] 
#attributes(incomplete)$variable.labels <- attr[,2]
nrow(incomplete)
nrow(data)
nrow(cleandf) 

# check amount of time needed to complete questionnaire
summary(cleandf[,118])
str(cleandf[,118])

# straightlining participants in G02?
straightl2 <- c()
for (i in 1:nrow(cleandf)){
  row <- cleandf[i,]
  if(all(c("Very pleasant", "Very convenient","Very safe", "Very important", "Very helpful", "Strongly agree", "Strongly agree") == cleandf[i,58:64])){
    print(paste("ID:", cleandf[i,1], "in row", i, "straightlined in QG 2a."))
    straightl2 <- append(straightl2, i)
  } else if(all(c("Pleasant", "Convenient", "Safe", "Important", "Helpful", "Agree", "Agree") == cleandf[i,58:64])){
    print(paste("ID:", cleandf[i,1], "in row", i, "straightlined in QG 2b."))
    straightl2 <- append(straightl2, i)
  } else if(all(c("Somewhat pleasant", "Somewhat convenient", "Somewhat safe", "Somewhat important", "Somewhat helpful", "Somewhat agree", "Somewhat agree") == cleandf[i,58:64])){
    print(paste("ID:", cleandf[i,1], "in row", i, "straightlined in QG 2c."))
    straightl2 <- append(straightl2, i)
  } else if(all(c("Neutral", "Neutral", "Neutral", "Neutral", "Neutral", "Neutral", "Neutral") == cleandf[i,58:64])){
    print(paste("ID:", cleandf[i,1], "in row", i, "straightlined in QG 2d."))
    straightl2 <- append(straightl2, i)
  } else if(all(c("Somewhat unpleasant", "Somewhat inconvenient", "Somewhat unsafe", "Somewhat unimportant", "Somewhat unhelpful", "Somewhat disagree", "Somewhat disagree") == cleandf[i,58:64])){
    print(paste("ID:", cleandf[i,1], "in row", i, "straightlined in QG 2e."))
    straightl2 <- append(straightl2, i)
  } else if(all(c("Unpleasant", "Inconvenient", "Unsafe", "Unimportant", "Unhelpful", "Disagree", "Disagree") == cleandf[i,58:64])){
    print(paste("ID:", cleandf[i,1], "in row", i, "straightlined in QG 2f."))
    straightl2 <- append(straightl2, i)
  } else if(all(c("Very unpleasant", "Very inconvenient", "Very unsafe", "Very unimportant", "Very unhelpful", "Strongly disagree", "Strongly disagree") == cleandf[i,58:64])){
    print(paste("ID:", cleandf[i,1], "in row", i, "straightlined in QG 2g."))
    straightl2 <- append(straightl2, i)
  }
}  
print(paste(length(straightl2), "participants straightlined in question group 2."))

# straghtlining participants in G03?
StrAgree <- as.data.frame(rowSums(cleandf[,65:89]=="Strongly agree"))
Agree <- as.data.frame(rowSums(cleandf[,65:89]=="Agree"))
SomeAgree <- as.data.frame(rowSums(cleandf[,65:89]=="Somewhat agree"))
Nand <- as.data.frame(rowSums(cleandf[,65:89]=="Neither agree nor disagree"))
Somedisagr <- as.data.frame(rowSums(cleandf[,65:89]=="Somewhat disagree"))                 
Disagr <- as.data.frame(rowSums(cleandf[,65:89]=="Disagree"))
Strdisagr <- as.data.frame(rowSums(cleandf[,65:89]=="Strongly disagree"))

straightl3 <- c()
for (i in 1:nrow(cleandf)){
  row <- cleandf[i,]
  if((StrAgree[i,] | Agree[i,] | SomeAgree[i,] | Nand[i,] | Somedisagr[i,] | Disagr[i,] | Strdisagr[i,]) == 25){
    print(paste("ID", cleandf[i,1], "in row", i, "straightlined in QG 3."))
    straightl3 <- append(straightl3, i)
  }
}  
print(paste(length(straightl3), "participants straightlined in question group 3."))

# straghtlining participants in G04?
straightl4a <- filter(cleandf, G04Q16_SQ001 == G04Q16_SQ002 & G04Q16_SQ002 == G04Q16_SQ003 & G04Q16_SQ003 == G04Q16_SQ004)
print(paste(nrow(straightl4a), "participants straightlined in question group 4, part 1."))

straightl4b <- filter(cleandf, 
                      G04Q17_SQ001 == G04Q17_SQ002 & G04Q17_SQ002 == G04Q17_SQ003 & 
                        G04Q17_SQ003 == G04Q17_SQ004 & G04Q17_SQ004 == G04Q17_SQ005 &
                        G04Q17_SQ005 == G04Q17_SQ006 & G04Q17_SQ006 == G04Q17_SQ007)
print(paste(nrow(straightl4b), "participants straightlined in question group 4, part 2."))
print(paste(nrow(intersect(straightl4a, straightl4b)), "participants straightlined in both parts of question group 4."))

# G05Q23: match "other" educational degrees to respective groups 
education <- keep(cleandf$G05Q23_other, is.na(cleandf$G05Q23_other)==FALSE)
education
cleandf$G05Q23_other <- str_replace_all(cleandf$G05Q23_other, ".*Fachabitur.*|.*Ausbildung.*|.*Knx.*|.*Bergbau.*|.*military.*|HS and tech|.*echnical.*|.*college.*|.*rade.*|automotive.*", "High School")
cleandf$G05Q23_other <- str_replace_all(cleandf$G05Q23_other, ".*Meister.*|.*meister.*|.*FH.*|.*HTL.*|College.*", "Bachelor's degree")
cleandf$G05Q23_other <- str_replace_all(cleandf$G05Q23_other, "^Dipl.*|zwei Master.*", "Master's degree")

for (i in 1:nrow(cleandf)){
  row <- cleandf[i,]
  if(cleandf$G05Q23[i]=="Other"){
    cleandf$G05Q23[i] <- cleandf$G05Q23_other[i]
  }
}  
unique(cleandf$G05Q23)

# G05Q24: match "other" occupations to respective groups
occupation <- keep(cleandf$G05Q24_other, is.na(cleandf$G05Q24_other)==FALSE)
unique(occupation)
cleandf$G05Q24_other <- str_replace_all(cleandf$G05Q24_other, "Unternehmer|Privatier.*", "Self-employed")
cleandf$G05Q24_other <- str_replace_all(cleandf$G05Q24_other, "Azubi", "Student")
cleandf$G05Q24_other <- str_replace_all(cleandf$G05Q24_other, "Disabled", "Unemployed")

for (i in 1:nrow(cleandf)){
  row <- cleandf[i,]
  if(cleandf$G05Q24[i]=="Other"){
    cleandf$G05Q24[i] <- cleandf$G05Q24_other[i]
  }
}  
unique(cleandf$G05Q24)

# G05Q27: match "other" occupation answers to respective groups
owner <- keep(cleandf$G05Q27_other, is.na(cleandf$G05Q27_other)==FALSE)
owner
cleandf$G05Q27_other <- str_replace_all(cleandf$G05Q27_other, ".*igentümer.*", "home owner")
cleandf$G05Q27_other <- str_replace_all(cleandf$G05Q27_other, ".*family.*|.*tudent.*|Bewohner", "renter")
unique(cleandf$G05Q27_other)

droprows <- c()
for (i in 1:nrow(cleandf)){
  row <- cleandf[i,]
  if(cleandf$G05Q27[i]=="Other"){
    cleandf$G05Q27[i] <- cleandf$G05Q27_other[i]
  }
  if(is.na(cleandf$G05Q27[i])==TRUE){
    droprows <- append(droprows,i)
  }
}  

cleandf <- cleandf[-c(droprows),] 
unique(cleandf$G05Q27)

# G01Q06: match "other" answers to respective groups --> can be dropped
household <- keep(cleandf$G01Q06_other, is.na(cleandf$G01Q06_other)==FALSE)
household

# G01Q07: does the sum equal 100 points?
sumres <- as.data.frame(cleandf[,47:56], stringsAsFactors=FALSE)
sumres[is.na(sumres)] <- 0
sapply(sumres,class)
sumres <- as.data.frame(apply(sumres, 2, as.numeric))  # Convert all variable types to numeric
sapply(sumres, class)  
totalsum <- transmute(rowwise(sumres), total=sum(c_across(1:10)))

if(sum(totalsum==100)==nrow(sumres)){
  print(paste("Each row equals 100 points."))
} else { print("There are some rows that do not equal 100 points.")
}

# G01Q08
reasons <- keep(cleandf$G01Q08, is.na(cleandf$G01Q08)==FALSE)
reasons

# G01Q03: merge G01Q03 and G01Q03_other
for (i in 1:nrow(cleandf)){
  row <- cleandf[i,]
  if(cleandf$G01Q03[i]=="Other"){
    cleandf$G01Q03[i] <- cleandf$G01Q03_other[i]
  }
} 
unique(cleandf$G01Q03)
#view(cleandf)

# drop columns that are not needed anymore: submitdate, G01Q03_other, G01Q06_other, G05Q23_other, G05Q24_other, G05Q27_other, Total time
cleandf <- cleandf[,-c(2,7,46,107,109,113,118)]
str(cleandf[,101:110])

# Relevel ordinal variable
cleandf$G05Q26 <- fct_relevel(cleandf$G05Q26, c("Rural area", "Town/semi-dense area", "City"))
cleandf$G05Q23 <- fct_relevel(cleandf$G05Q23, c("Elementary School", "Middle School", "High School", "Bachelor's degree", "Master's degree", "Doctoral degree/advanced degree"))
cleandf$G05Q28 <- fct_relevel(cleandf$G05Q28, c("I live alone.", "Two people", "3-4 people", "5-6 people", "7+ people"))
cleandf$G01Q02 <- fct_relevel(cleandf$G01Q02, c("less than six months", "6-12 months", "1-3 years", "4-6 years", "7-10 years", "more than 10 years"))

# ----- ANALYZING INCOMPLETE DATA -----
# checking incomplete data to identify potential patterns
print(paste((nrow(incomplete)-(sum(incomplete$G01Q01=="Yes", na.rm=TRUE))), "entries were removed because these participants were not current smart home users."))
incomplete <- incomplete[c(incomplete$G01Q01=="Yes", na.rm=TRUE),]
dropinc <- c()
for (j in 1:nrow(incomplete)){
  row <- incomplete[j,]
  if((is.na(incomplete[j,3]))){
    dropinc <- append(dropinc, j)
  } 
}
incomplete <- incomplete[-c(unique(dropinc)),]
nrow(incomplete)
str(incomplete)
summary(incomplete)

demdata <- c()
for (i in 1:nrow(incomplete)){
  row <- incomplete[i,]
  if(sum(is.na(incomplete[i,104:116]))>12){
    demdata <- append(demdata, i)
    print(paste("Id:", incomplete[i,1], " in row", i, "has no entries for sociodemographic data."))
  }
}
incomplete <- incomplete[-c(unique(demdata)),]
print(paste((nrow(as.matrix(demdata))), "entries were removed due to missing socio-demographic data."))

modeldata <- c()
for (i in 1:nrow(incomplete)){
  row <- incomplete[i,]
  if(sum(is.na(incomplete[i,58:100]))>5){
    modeldata <- append(modeldata, i)
    print(paste("Id:", incomplete[i,1], " in row", i, "has more than 5 NA entries for the theoretical model."))
  }
}
incomplete <- incomplete[-c(unique(modeldata)),]
print(paste((nrow(as.matrix(modeldata))), "entries were removed due to lack of data for the theoretical model."))
#print(paste(nrow(incomplete), "entries remaining in the incomplete data."))

# ==============================================================================================================================
# =====  DESCRIPTIVE STATISTICS  
# Create table to depict overview of descriptive data
source("functions.R")
descstat <- dstat(cleandf)

collapse_rows_dt <- data.frame(Variable=c(rep("Gender", 3), 
                                          rep("Age", 6), 
                                          rep("Education", 6), 
                                          rep("Occupation status", 6),
                                          rep("Country", 6),
                                          rep("Area population", 3),
                                          rep("Housing", 2),
                                          rep("Household size", 5),
                                          rep("Children per household", 6),
                                          rep("Household income per month", 6),
                                          rep("Total (n)",1)),
                               Attribute=c(descstat$Category, ""),
                               Frequency=c(descstat$Frequency, nrow(cleandf)),
                               Proportion=c(descstat$Proportion, 1))

kbl(collapse_rows_dt, booktabs=T, caption="Descriptive Statistics of Survey Participants", align = "c", format="html", escape=TRUE) %>%
    kable_styling(bootstrap_options=c("condensed","bordered"), position="center", font_size=10, full_width=F, html_font="arial") %>%
  row_spec(c(1:3,10:15,22:27,31:32,38:43,50), hline_after=T, background="#F1F1F1" ) %>%
  row_spec(c(2,6,13,16,22,30,31,35,38,46,50), bold=T) %>%
  column_spec(1, bold = T) %>%
  collapse_rows(columns = 1, valign = "middle") %>%
  save_kable(file="table1.html", self_contained=F)

# Categorical variables
categ <- descstat[c(1:3,16:27,31,32),2:5] %>%
  group_by(Variable) %>%
  arrange(desc(Frequency), .by_group=TRUE)

categorical <- data.frame(Variable=c(rep("Gender", 3), 
                                     rep("Occupation status", 6),
                                     rep("Country", 6),
                                     rep("Housing", 2),
                                     rep("Total (n)",1)),
                          Attribute=c(categ$Category, ""),
                          Frequency=c(categ$Frequency, nrow(cleandf)),
                          Proportion=c(categ$Proportion, 1))

webshot::install_phantomjs()
kbl(categorical, booktabs=T, caption="Categorical Variables", align = "c", format="html", escape=TRUE) %>%
  kable_styling(bootstrap_options=c("condensed","bordered"), position="center", font_size=10, full_width=F, html_font="arial") %>%
  row_spec(c(1:3,10:15,18), hline_after=T, background="#F1F1F1" ) %>%
  column_spec(1, bold = T) %>%
  collapse_rows(columns = 1, valign = "middle") %>%
  save_kable(file="categ.png", zoom=1.5, self_contained=F)

# Ordinal variables: age, education, household size, children per household, income 
ordin <- descstat[c(4:15,28:30,33:49),2:5] %>%
  group_by(Variable) %>%
  mutate("Cumulative Frequency"=cumsum(Frequency)) %>%
  mutate("Cumulative Proportion"=cumsum(Proportion))

ordinal <- data.frame(Variable=c(rep("Age", 6), 
                                 rep("Education", 6), 
                                 rep("Area population", 3),
                                 rep("Household size", 5),
                                 rep("Children per household", 6),
                                 rep("Household income per month", 6)),
                      Attribute=c(ordin$Category),
                      Frequency=c(ordin$Frequency),
                      Proportion=c(ordin$Proportion),
                      "Cumul Freq"=c(ordin$`Cumulative Frequency`),
                      "Cumul Prop"=c(ordin$`Cumulative Proportion`))

kbl(ordinal, booktabs=T, caption="Ordinal Variables", align = "c", format="html",escape=TRUE) %>%
  kable_styling(bootstrap_options=c("condensed","bordered"), position="center", full_width=F,font_size=10, html_font="arial") %>%
  row_spec(c(1:6,13:15,21:26), hline_after=T, background="#F1F1F1" ) %>%
  row_spec(c(3,10,15,18,21,29), bold=T) %>%
  column_spec(1, bold = T, width="3cm") %>%
  column_spec(2, width="4.5cm") %>%
  collapse_rows(columns = 1, valign = "middle") %>%
  save_kable(file="ordin.jpeg", zoom=1.5, self_contained=F)

# Smart home information
dfSH <- cleandf[,c(3:54, 91:97)] 
dfSH <- as.data.frame(unclass(dfSH), stringsAsFactors=TRUE)
sapply(dfSH,class)
str(dfSH)

# Length of use
summary(dfSH$G01Q02)

# SH system name
shsystems <- as.data.frame(unique(dfSH$G01Q03))
shsystems2 <- as.data.frame(summary(dfSH$G01Q03))
#view(shsystems2)
shnames <- as.vector(rownames(shsystems))
print(shsystems)
dfSH$G01Q03

sum(str_count(dfSH$G01Q03, pattern=c(".*mazon.*|.*lexa*.|.*echo*.")))
sum(str_count(dfSH$G01Q03, pattern=c(".*omekit*.|.*omeKit*.|.*omebridge*.")))
sum(str_count(dfSH$G01Q03, pattern=c(".*oogle*.")))
sum(str_count(dfSH$G01Q03, pattern=c(".*omematic*.|.*omatic*.|.*hmip*.")))
sum(str_count(dfSH$G01Q03, pattern=c(".*ssistant*.")))
sum(str_count(dfSH$G01Q03, pattern=c(".*broker*.")))
sum(str_count(dfSH$G01Q03, pattern=c(".*penHAB*.")))
sum(str_count(dfSH$G01Q03, pattern=c(".*knx*.|.*KNX*.")))
sum(str_count(dfSH$G01Q03, pattern=c(".*bosch*.|.*Bosch*.")))
sum(str_count(dfSH$G01Q03, pattern=c(".*hue*.|.*Hue.*")))
sum(str_count(dfSH$G01Q03, pattern=c(".*cobee*.")))
sum(str_count(dfSH$G01Q03, pattern=c(".*ikea*.|.*radfri*.")))
sum(str_count(dfSH$G01Q03, pattern=c(".*ubitat*.|.*abitat.*")))

# SHEMS as part of SH
summary(dfSH$G01Q04)

# Number of SH products
products <- cleandf[,7:41]
# Number of counts per product category
prodcount <- sapply(products, function(x) table(factor(x, ordered=TRUE)))
prodcount <- as.matrix(prodcount[-1,])
rownames(prodcount)
prodnames <- c("Smart thermostat", "Smart plugs", "Smart meter", "Heating system", "Solar panels", "Energy storage system", "Light sources", "AC", "Ventilation", "Water heater", "Shading devices", "Switches", "Garage door controls", "Door locks", "Motion sensors", "Door and window sensors", "Environmental sensors", "Flood sensors", "Fire, smoke, or gas detection", "Security cameras", "Refrigerator", "Stove/oven", "Dishwasher", "Washing machine", "Dryer", "Coffee machine", "Microwave", "Vacuum cleaner", "Mowing robot", "Smart TV", "Sound system", "Smart speaker", "Streaming devices", "Health-related devices", "Electric vehicle")
rownames(prodcount) <- prodnames
colnames(prodcount) <- c("Count")
print(prodcount)
unique(dfSH$G01Q05_other)
prodcount <- as.data.frame(prodcount)
prodcount$Percentage <- round((prodcount$Count/363)*100, digits=2)
colnames(prodcount) <- c("Count", "Percentage %")
prodcount <- prodcount[order(-prodcount[,"Percentage %"]),]
formattable(prodcount)

# Number of products per survey participant
levels <- c("Yes","N/A")
prodpart <- sapply(levels,function(x)rowSums(products==x)) #count occurrences of x in each row
colnames(prodpart) <- levels
prodpart <- prodpart[,-2]
#view(prodpart)
summary(prodpart)
hist(prodpart, main="Products per Participant", xlab="Number of Products")

# Main responsible person within the household
summary(dfSH$G01Q06)

# Importance of reasons to install SHT
importance <- as.data.frame(sapply(cleandf[,44:53],as.numeric))
colnames(importance) <- c("Interest", "Sustainability", "Entertainment", "Comfort", "Money", "AAL", "Safety", "Control", "Pre-installed", "Other")
averageimp <- sapply(importance, function(x) mean(x,na.rm=TRUE))
sumimp <- sapply(importance, function(x) sum(x,na.rm=TRUE))
maxColumn <- names(sumimp)[which.max(sumimp)]
maxPoints <- max(sumimp,na.rm=TRUE)
minColumn <- names(sumimp)[which.min(sumimp)]
minPoints <- min(sumimp,na.rm=TRUE)
print("Average Importance:")
print(sort(round(averageimp,2)), decreasing=TRUE)
print(paste("The most important reason to install SH was: ", maxColumn, "(with an importance of", maxPoints, ")"))
print(paste("The least important reason to install SH was: ", minColumn, "(with an importance of", minPoints,")"))

importanceTable <- st(importance,summ=c('mean(x)'),out='return')
importanceTable$Mean <- as.numeric(importanceTable$Mean)
importanceTable <- importanceTable[order(-importanceTable$Mean),]
rownames(importanceTable) <- 1:10
importanceTable$Importance <- row.names(importanceTable)
#view(unique(dfSH$G01Q08))

# SH Feedback
SHfeedback <- cleandf[,91:97]
SHfeedback <- as.data.frame(unclass(SHfeedback), stringsAsFactors=TRUE)
colnames(SHfeedback) <- c("Easier & personalized", "Comparative consumption", "Goal setting", "Savings", "Rewards", "Env. effect", "Gamified")
summary(SHfeedback)

mostFreqResponse <- sapply(SHfeedback, function(x) names(which.max(table(factor(x, ordered=TRUE)))))
levels <- c("Very  likely","Already  in use")
mostPosResponse <- sapply(levels, function(x) colSums(SHfeedback==x))
mostPosResponse_Sum <- rowSums(mostPosResponse)
mostPosColumn <- names(mostPosResponse_Sum)[which.max(mostPosResponse_Sum)]
print("Most Frequent Response per incentive:")
print(mostFreqResponse)
print(paste("Most Positive Column: ", mostPosColumn))
print(mostPosResponse)

mostPosResponse <- as.data.frame(mostPosResponse)
mostPosResponse$MostFreqResp <- mostFreqResponse
colnames(mostPosResponse) <- c('Very likely',"Already in use", "Most frequent response")
mostPosResponse <- mostPosResponse[order(-mostPosResponse_Sum),]
mostPosResponse <- mostPosResponse[,order(c("Most frequent response","Very likely", "Already in use"))]

# Energy bills + consumption
print("Have you noticed any cost savings on your energy bills since installing your smart home system?")
energyBills <- as.data.frame(summary(as.factor(cleandf$G04Q18)))
colnames(energyBills) <- c("No. of answers")
energyBills$Percentage <- (energyBills[,1]/sum(energyBills[,1]))*100
colnames(energyBills) <- c("No. of answers","Percentage %")
formattable(energyBills,digits=4)

print("Since using my smart home, my energy consumption ??")
consumption <- as.data.frame(summary(as.factor(cleandf$G04Q19)))
colnames(consumption) <- c("No. of answers")
consumption$Percentage <- (consumption[,1]/sum(consumption[,1]))*100
colnames(consumption) <- c("No. of answers","Percentage %")
formattable(consumption,digits=4)

energyBills$Group <- "Have you noticed any costs savings on your energy bills since installing SH?"
consumption$Group <- "My energy consumption ..."
groupTable <- rbind(energyBills, consumption)
groupTable$Answer <- rownames(groupTable)
groupTable <- groupTable[,c("Group","Answer","No. of answers", "Percentage %")]
groupTable %>% mutate_at(vars(4), funs(round(., 1))) %>% group_by(Group) %>% gt(rowname_col="crop")

# Usability
cleandf$G04Q20 <- fct_relevel(cleandf$G04Q20, c("Very poor", "Poor", "Okay", "Good", "Very good"))
print("How would you rate the overall usability of your smart home system and/or smart home energy management system?")
usability <- as.data.frame(summary(cleandf$G04Q20))
usability$Percentage <- (usability[,1]/sum(usability[,1]))*100
colnames(usability) <- c("No. of answers","Percentage %")
formattable(usability,digits=2)

## ----- World cloud -----
feedb_en <- cleandf[which(cleandf$startlanguage=="en" & !is.na(cleandf$G05Q31),arr.ind=TRUE), "G05Q31"]
feedb_de <- cleandf[which(cleandf$startlanguage=="de" & !is.na(cleandf$G05Q31),arr.ind=TRUE), "G05Q31"]
feedb_en <- Corpus(VectorSource(feedb_en))
feedb_de <- Corpus(VectorSource(feedb_de))
feedb_en <- feedb_en %>%
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace) 
feedb_de <- feedb_de %>%
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace) 
feedb_en <- tm_map(feedb_en, content_transformer(tolower))
feedb_de <- tm_map(feedb_de, content_transformer(tolower))
feedb_en <- tm_map(feedb_en, removeWords, stopwords("english"))
feedb_de <- tm_map(feedb_de, removeWords, stopwords("german"))
dtmen <- TermDocumentMatrix(feedb_en)
dtmde <- TermDocumentMatrix(feedb_de)
wc_en <- as.matrix(dtmen)
wc_de <- as.matrix(dtmde)
words_en <- sort(rowSums(wc_en), decreasing=TRUE)
words_de <- sort(rowSums(wc_de), decreasing=TRUE)
words_en
wordcloud_en <- data.frame(word=names(words_en), freq=words_en)
wordcloud_de <- data.frame(word=names(words_de), freq=words_de)
set.seed(1234)
wordcloud_en <- wordcloud(words=wordcloud_en$word, freq=wordcloud_en$freq, min.freq=1, max.words=200, random.order=FALSE, rot.per=0.35, colors=brewer.pal(8,"Dark2"))
set.seed(1234)
wordcloud_de <- wordcloud(words=wordcloud_de$word, freq=wordcloud_de$freq, min.freq=1, max.words=200, random.order=FALSE, rot.per=0.35, colors=brewer.pal(8,"Dark2"))

# Data frame for items 
df2 <- cleandf
source("functions.R")
df2 <- syntax(df2)
#sum(is.na(df2[,55:90]))
df2 <- df2[,55:90]
indx <- sapply(df2, is.factor)
df2[indx] <- lapply(df2[indx], function(x) as.numeric(as.character(x)))

# Reversing scores for PBC5
reverseitem <- which(colnames(df2)=="PBC5")
newdf <- as.data.frame(reverse.code(reverseitem,df2))
colnames(newdf)[17] <- "PBC5"
#colnames(newdf)

# create df for consolidated attributes
condf <- as.data.frame(newdf) %>%
  rowwise() %>%
  mutate(AT=mean(c(AT1,AT2,AT3,AT4,AT5,AT6,AT7))) %>%
  mutate(SN=mean(c(SN1,SN2,SN3,SN4,SN5)))  %>%
  mutate(PBC=mean(c(PBC1,PBC2,PBC3,PBC4,PBC5,PBC6))) %>%
  mutate(AC=mean(c(AC1,AC2,AC3))) %>%
  mutate(AR=mean(c(AR1,AR2,AR3))) %>%
  mutate(PN=mean(c(PN1,PN2,PN3))) %>%
  mutate(BI=mean(c(BI1,BI2,BI3,BI4,BI5))) %>%
  mutate(PB=mean(c(PB1,PB2,PB3,PB4)))
condf <- condf[-c(1:36)]

# ==================================================================================================================================

# =====  STRUCTURAL EQUATION MODELING
newdf <- as_tibble(newdf)

# data distribution
mardiaSkew(newdf, use="everything")
mardiaKurtosis(newdf, use="everything")
itemstat <- as.list(describe(newdf))
distribution <- bind_rows(lapply(newdf, function(x) as.vector(shapiro_test(x))),.id="variable")
distribution[,2:3] <- round(distribution[,2:3],2)

mshapiro_test(newdf)
stats <- bind_cols(list(itemstat, distribution))
stats <- stats[-c(1,2,5:10,13)]
stats <- relocate(stats, variable, .before=mean)
stats <- rename(stats, Indicator=variable, Mean=mean, Std.=sd, Skew=skew, Kurtosis=kurtosis, Statistic=statistic)
# indication for multivariate non-normal data 

# Measurement model
# excluding moderator
mmod1a <- constructs(
  composite("AT", multi_items("AT", 1:7)),
  composite("SN", multi_items("SN", 1:5)),
  composite("PBC", multi_items("PBC", 1:6)),
  composite("AC", multi_items("AC", 1:3)),
  composite("AR", multi_items("AR", 1:3)),
  composite("PN", multi_items("PN", 1:3)),
  composite("BI", multi_items("BI", 1:5))
)

# no moderator, excluding AT1, AT2, AT3, AT6, SN4, SN5, PBC2, PBC3, PBC5, PBC6, BI5  
mmod1b <- constructs(
  composite("AT", multi_items("AT", c(4:5,7))),
  composite("SN", multi_items("SN", 1:3)),
  composite("PBC", multi_items("PBC", c(1,4))),
  composite("AC", multi_items("AC", 1:3)),
  composite("AR", multi_items("AR", 1:3)),
  composite("PN", multi_items("PN", 1:3)),
  composite("BI", multi_items("BI", 1:4))
)

# Measurement model including moderator
mmod2a <- constructs(
  composite("AT", multi_items("AT", 1:7)),
  composite("SN", multi_items("SN", 1:5)),
  composite("PBC", multi_items("PBC", 1:6)),
  composite("AC", multi_items("AC", 1:3)),
  composite("AR", multi_items("AR", 1:3)),
  composite("PN", multi_items("PN", 1:3)),
  composite("BI", multi_items("BI", 1:5)),
  composite("PB", multi_items("PB", 1:4)),
  interaction_term(iv="AT", moderator="PB", method=two_stage),
  interaction_term(iv="SN", moderator="PB", method=two_stage),
  interaction_term(iv="PN", moderator="PB", method=two_stage),
  interaction_term(iv="PBC", moderator="PB", method=two_stage)
)

# excluding AT1, AT2, AT3, AT6, SN4, SN5, PBC2, PBC3, PBC5, PBC6, PB2, PB4, BI5  
mmod2c <- constructs(
  composite("AT", multi_items("AT", c(4:5,7))),
  composite("SN", multi_items("SN", 1:3)),
  composite("PBC", multi_items("PBC", c(1,4))),
  composite("AC", multi_items("AC", 1:3)),
  composite("AR", multi_items("AR", 1:3)),
  composite("PN", multi_items("PN", 1:3)),
  composite("BI", multi_items("BI", 1:4)),
  composite("PB", multi_items("PB", c(1,3))),
  interaction_term(iv="AT", moderator="PB", method=two_stage),
  interaction_term(iv="SN", moderator="PB", method=two_stage),
  interaction_term(iv="PN", moderator="PB", method=two_stage),
  interaction_term(iv="PBC", moderator="PB", method=two_stage)
)

mmod2d <- constructs(
  composite("AT", multi_items("AT", c(4:5,7))),
  composite("SN", multi_items("SN", 1:3)),
  composite("PBC", multi_items("PBC", c(1,4))),
  composite("AC", multi_items("AC", 1:3)),
  composite("AR", multi_items("AR", 1:3)),
  composite("PN", multi_items("PN", 1:3)),
  composite("BI", multi_items("BI", 1:4)),
  composite("PB", multi_items("PB", c(1,3))),
  interaction_term(iv="PN", moderator="PB", method=two_stage))

# Structural model without moderator
strucmod1a <- relationships(
  paths(from=c("AC"), to = c("AR","AT")),
  paths(from=c("AR", "SN"), to = c("PN")),
  paths(from=c("AT","PBC", "SN", "PN"), to = c("BI"))
)

# Structural model with moderator
strucmod2a <- relationships(
  paths(from=c("AC"), to = c("AR","AT")),
  paths(from=c("AR", "SN"), to = c("PN")),
  paths(from=c("AT","PBC", "SN", "PN", "PB", "AT*PB", "SN*PB", "PN*PB", "PBC*PB"), to = c("BI"))
)

# Structural model without AC->AR but AC->PN
strucmod2b <- relationships(
  paths(from=c("AC"), to = c("AT")),
  paths(from=c("AC", "AR", "SN"), to = c("PN")),
  paths(from=c("AT","PBC", "SN", "PN", "PB", "AT*PB", "SN*PB", "PN*PB", "PBC*PB"), to = c("BI"))
)

# Structural model with moderator for only PN
strucmod2f <- relationships(
  paths(from=c("AC"), to = c("AR","AT")),
  paths(from=c("AR", "SN"), to = c("PN")),
  paths(from=c("AT","PBC", "SN", "PN", "PB", "PN*PB"), to = c("BI"))
)

# Model estimation using PLS-SEM algorithm
modfit1a <- estimate_pls(data=newdf, measurement_model=mmod1a, structural_model=strucmod1a)
modfit1b <- estimate_pls(data=newdf,measurement_model=mmod1b,structural_model=strucmod1a)
modfit1c <- estimate_pls(data=newdf,measurement_model=mmod1b,structural_model=strucmod2b)
modfit2a <- estimate_pls(data=newdf, measurement_model=mmod2a, structural_model=strucmod2a)
modfit2c <- estimate_pls(data=newdf, measurement_model=mmod2c, structural_model=strucmod2a)
modfit2d <- estimate_pls(data=newdf, measurement_model=mmod2d, structural_model=strucmod2f)

# Model summary
modsum1a <- summary(modfit1a)
modsum1b <- summary(modfit1b)
modsum1c <- summary(modfit1c)
modsum2a <- summary(modfit2a)
modsum2c <- summary(modfit2c)
modsum2d <- summary(modfit2c)

# --- Reflective measurement model evaluation
# Indicator reliability
# Iterations to converge
modsum1a$iterations
modsum1b$iterations
modsum2a$iterations
modsum2c$iterations

# Indicator loadings >= 0.708
modsum1a$loadings
modsum1b$loadings
modsum2a$loadings
modsum2c$loadings

# Indicator reliability >= 0.5
modsum1a$loadings^2
modsum1b$loadings^2
modsum2a$loadings^2
modsum2c$loadings^2

# Internal consistency reliability 
# Cronbach's alpha, 0.7>= rho <= 0.9 
# Convergent validity - AVE >= 0.5
modsum1a$reliability
modsum1b$reliability
modsum2a$reliability
modsum2c$reliability
#plot(modsum1a$reliability[,c("alpha", "rhoC", "rhoA")])

#Discriminant validity - HTMT criterion <0.9
modsum1a$validity$htmt
modsum1b$validity$htmt
modsum1c$validity$htmt
modsum2a$validity$htmt
modsum2c$validity$htmt

# Bootstrapping for significance test
bootmodev1a <- bootstrap_model(seminr_model=modfit1a, nboot=10000)
bootmodev1b <- bootstrap_model(seminr_model=modfit1b, nboot=10000)
bootmodev2a <- bootstrap_model(seminr_model=modfit2a, nboot=10000)
bootmodev2c <- bootstrap_model(seminr_model=modfit2c, nboot=10000)
bootmodev2d <- bootstrap_model(seminr_model=modfit2d, nboot=10000)
sumbootmodev1a <- summary(bootmodev1a, alpha = 0.05)
sumbootmodev1b <- summary(bootmodev1b, alpha = 0.05)
sumbootmodev1bhtmt <- summary(bootmodev1b, alpha = 0.10)
sumbootmodev2a <- summary(bootmodev2a, alpha = 0.05)
sumbootmodev2c <- summary(bootmodev2c, alpha = 0.05)
sumbootmodev2d <- summary(bootmodev2d, alpha = 0.05)

# Extract bootstrapped HTMT
sumbootmodev1a$bootstrapped_HTMT
sumbootmodev1b$bootstrapped_HTMT
sumbootmodev2a$bootstrapped_HTMT
sumbootmodev2c$bootstrapped_HTMT
sumbootmodev1bhtmt$bootstrapped_HTMT

# --- Structural model evaluation
# Collinearity of the structural model - VIF<3
modsum1a$vif_antecedents
modsum1b$vif_antecedents
modsum2a$vif_antecedents
modsum2c$vif_antecedents

# Significance and relevance of path coefficients
sumbootmodev1a$bootstrapped_paths 
sumbootmodev1b$bootstrapped_paths 
sumbootmodev2a$bootstrapped_paths 
sumbootmodev2c$bootstrapped_paths 
sumbootmodev2d$bootstrapped_paths 

# Inspect the total effects
sumbootmodev1a$bootstrapped_total_paths 
sumbootmodev1b$bootstrapped_total_paths 
sumbootmodev2a$bootstrapped_total_paths 
sumbootmodev2c$bootstrapped_total_paths 
#plot(summary(modfit2a)$reliability)

# Explanatory power: model RSquares, 0.75, 0.5, 0.25 == substantial, moderate, weak
# path coefficients and r^2 values
modsum1a$paths 
modsum1b$paths 
modsum2a$paths 
modsum2c$paths 

# Inspect the effect sizes
modsum1b$fSquare
modsum2a$fSquare
modsum2c$fSquare

# Predictive power
# Generate model predictions
predict1a <- predict_pls(model = modfit1a, technique=predict_DA, noFolds=5, reps=5)
predict1b <- predict_pls(model = modfit1b, technique=predict_DA, noFolds=5, reps=5)
predict2a <- predict_pls(model = modfit2a, technique=predict_DA, noFolds=5, reps=5)
predict2c <- predict_pls(model = modfit2c, technique=predict_DA, noFolds=5, reps=5)

# Summarize prediction results
sumpredict1a <- summary(predict1a)
sumpredict1b <- summary(predict1b)
sumpredict2a <- summary(predict2a)
sumpredict2c <- summary(predict2c)

# Analyze the distribution of prediction error
par(mfrow=c(1,4))
plot(sumpredict1b, indicator = "BI1")
plot(sumpredict1b, indicator = "BI2")
plot(sumpredict1b, indicator = "BI3")
plot(sumpredict1b, indicator = "BI4")
par(mfrow=c(1,1))

# Prediction statistics
sumpredict1a 
sumpredict1b 
sumpredict2a 
sumpredict2c 

# --- Model comparisons
# IT criteria
modsum1a$it_criteria
# Subset the matrix to only return the BIC row and CUSL column
modsum1a$it_criteria["BIC", "BI"]
# Collect the vector of BIC values 
itcriteriavector <- c(modsum1b$it_criteria["BIC","BI"],
                      modsum2c$it_criteria["BIC","BI"],
                      modsum2d$it_criteria["BIC", "BI"])
                   
# Inspect IT Criteria vector for competing models
itcriteriavector
# Calculate model BIC Akaike weights
compute_itcriteria_weights(itcriteriavector)


# --- Moderation analysis
sumbootmodev2a$bootstrapped_paths
sumbootmodev2c$bootstrapped_paths

# Simple slope analysis plot for AT
plotAT <- slope_analysis(moderated_model = modfit2c,
               dv = "BI",
               moderator = "PB",
               iv = "AT",
               leg_place = "bottomright")

# Simple slope analysis plot for SN
plotSN <- slope_analysis(moderated_model = modfit2c,
               dv = "BI",
               moderator = "PB",
               iv = "SN",
               leg_place = "bottomright")

# Simple slope analysis plot for PBC
plotPBC <- slope_analysis(moderated_model = modfit2c,
               dv = "BI",
               moderator = "PB",
               iv = "PBC",
               leg_place = "bottomright")

# Simple slope analysis plot for PN
plotPN <- slope_analysis(moderated_model = modfit2c,
               dv = "BI",
               moderator = "PB",
               iv = "PN",
               leg_place = "bottomright")

# Plot final Path model 
thm <- seminr_theme_create(plot.specialcharacters = TRUE,
                           mm.node.label.fontsize = 16,
                           mm.edge.label.fontsize=16,
                           mm.edge.boot.show_p_stars = TRUE,
                           sm.edge.boot.show_ci = FALSE,
                           sm.node.label.fontsize = 18,
                           sm.edge.boot.show_p_stars = TRUE,
                           sm.edge.label.fontsize = 16,
                           sm.edge.boot.show_p_value = FALSE,
                           sm.edge.boot.show_t_value = FALSE,
                           sm.edge.positive.color = "palegreen3",
                           sm.edge.negative.color = "indianred1",
                           sm.edge.label.show = TRUE,
                           mm.node.color = "snow3",
                           mm.node.fill="white",
                           sm.edge.boot.template= "{variable} = {value}{stars}<BR /><FONT POINT-SIZE='16'>{civalue} {tvalue} {pvalue} </FONT>",
                           mm.edge.positive.color="snow3",
                           mm.edge.negative.color="snow3",
                           mm.edge.positive.style="solid",
                           sm.node.fill = "lightcyan",
                           sm.edge.negative.style = "solid")

seminr_theme_set(thm)
thm <- seminr_theme_smart()
plot(bootmodev2c)
save_plot("finalmodel.png", width=1500, height=1200)

# --- Mediation effect SN -> PN
modsum1b$total_indirect_effects

# Inspect indirect effects
specific_effect_significance(bootmodev1b, from = "SN",
                           through = "PN",
                           to = "BI",
                           alpha = 0.05)
specific_effect_significance(bootmodev1b, from = "SN",
                             through = "PN",
                             to = "BI",
                             alpha = 0.05)

# Inspect direct effects 
modsum1b$paths

# Inspect confidence intervals for direct effects 
sumbootmodev1b$bootstrapped_paths

# Calculate the sign of p1*p2*p3
modsum1b$paths["SN", "PN"] *  modsum1a$paths["PN", "BI"] *  modsum1a$paths["SN", "BI"]
# --> complementary partial mediator

#paths <- sumbootmodev2c$bootstrapped_paths[, "Original Est."]
#t_value <- sumbootmodev2c$bootstrapped_paths[, "T Stat."]
#pvalues <- stats::pt(abs(t_value), nrow(bootmodev2c$data) - 1, lower.tail = FALSE) %>% 
#  round(digits = 3)
#pvalues

# excluding AT1, AT2, AT3, AT6, SN4, SN5, PBC2, PBC3, PBC5, PBC6, PB2, PB4, BI5 for Random forest 
dfexcl2 <- newdf[c("AT4", "AT5", "AT7", "SN1", "SN2", "SN3", "AC1", "AC2", "AC3", "AR1", "AR2", "AR3", "PN1", "PN2", "PN3", "BI1", "BI2", "BI3", "BI4", "PB1", "PB3")]
