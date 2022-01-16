# Name: Hee-Eun Lee ----- Student no: 558864 ----- Master's thesis
#-------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------- Functions ---------------------------------------------------------------

# Overview of functions:
# 1. Table for descriptive statistics
# 2. Syntax for items

#------------------------------------------------------------------------------------------------------------------------------------
# 1. Table for descriptive statistics:

dstat <- function(cleandf){
  cleandf <- relocate(cleandf, G05Q21, .after=G05Q22)
  
  cleandf$G05Q30 <- fct_expand(cleandf$G05Q30, c("less than 1,000 USD","up to 3,000 USD", "8,001-15,000 USD", "15,001-50,000 USD", "1,001-3,000 USD", "3,001-5,000 USD", "5,001-8,000 USD", "8,001-10,000 USD", "10,001-15,000 USD", "15,001-20,000 USD", "20,001-30,000 USD", "30,001-50,000 USD", "more than 50,000 USD"))
  for (i in 1:nrow(cleandf)){
    if(cleandf$G05Q30[i]==("less than 1,000 USD") || cleandf$G05Q30[i]==("1,001-3,000 USD")){
      cleandf$G05Q30[i] <- c("up to 3,000 USD")
    } else if 
    (cleandf$G05Q30[i]==("8,001-10,000 USD") || cleandf$G05Q30[i]==("10,001-15,000 USD")){
      cleandf$G05Q30[i] <- c("8,001-15,000 USD")
    } else if 
    (cleandf$G05Q30[i]==("15,001-20,000 USD") || cleandf$G05Q30[i]==("20,001-30,000 USD") || cleandf$G05Q30[i]==("30,001-50,000 USD")){
      cleandf$G05Q30[i] <- c("15,001-50,000 USD")
    }
  }
  
  cleandf$G05Q30 <- fct_drop(cleandf$G05Q30, c("less than 1,000 USD", "1,001-3,000 USD", "8,001-10,000 USD", "10,001-15,000 USD", "15,001-20,000 USD", "20,001-30,000 USD", "30,001-50,000 USD"))
  cleandf$G05Q30 <- fct_relevel(cleandf$G05Q30, c("up to 3,000 USD", "3,001-5,000 USD", "5,001-8,000 USD", "8,001-15,000 USD", "15,001-50,000 USD", "more than 50,000 USD"))
  
  for (i in 1:nrow(cleandf)){
    if(cleandf$G05Q29[i]=="3" || cleandf$G05Q29[i]=="4"){
      cleandf$G05Q29[i] <- c("3-4")
    }
  }
  
  cleandf[sapply(cleandf, is.character)] <- lapply(cleandf[sapply(cleandf, is.character)], 
                                                   as.factor)
  
  descstat <- summary(cleandf[,101:110], maxsum=6) %>%
    as.data.frame() %>%
    drop_na() %>%
    separate(col="Freq", sep=":", into=c("Category", "Frequency"))
  view(descstat)
  descstat$Frequency <- as.numeric(descstat$Frequency)
  descstat <- mutate(descstat, Proportion=percent(Frequency/nrow(cleandf)))
  
  rename(descstat, Variable=Var2)
}

#-------------------------------------------------------------------------------------------------------------------------------------
# Syntax for items
syntax <- function(df){
  df$G02Q09_SQ001 <- factor(as.factor(df$G02Q09_SQ001),levels=c("Very  unpleasant", "Unpleasant", "Somewhat  unpleasant", "Neutral", "Somewhat  pleasant", "Pleasant", "Very  pleasant"),labels=c("1","2","3","4","5","6","7")) 
  df$G02Q10_SQ001 <- factor(as.factor(df$G02Q10_SQ001),levels=c("Very  inconvenient", "Inconvenient", "Somewhat  inconvenient", "Neutral", "Somewhat  convenient", "Convenient", "Very  convenient"),labels=c("1","2","3","4","5","6","7")) 
  df$G02Q11_SQ001 <- factor(as.factor(df$G02Q11_SQ001),levels=c("Very  unsafe", "Unsafe", "Somewhat  unsafe", "Neutral", "Somewhat  safe", "Safe", "Very  safe"),labels=c("1","2","3","4","5","6","7")) 
  df$G02Q12_SQ001 <- factor(as.factor(df$G02Q12_SQ001),levels=c("Very  unimportant", "Unimportant", "Somewhat  unimportant", "Neutral", "Somewhat  important", "Important", "Very  important"),labels=c("1","2","3","4","5","6","7")) 
  df$G02Q13_SQ001 <- factor(as.factor(df$G02Q13_SQ001),levels=c("Very  unhelpful", "Unhelpful", "Somewhat  unhelpful", "Neutral", "Somewhat  helpful", "Helpful", "Very  helpful"),labels=c("1","2","3","4","5","6","7")) 
  df$G02Q14_SQ001 <- factor(as.factor(df$G02Q14_SQ001),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neutral", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7")) 
  df$G02Q14_SQ002 <- factor(as.factor(df$G02Q14_SQ002),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neutral", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7")) 
  df$G04Q16_SQ001 <- factor(as.factor(df$G04Q16_SQ001),levels=c("Never", "Very rarely", "Every  other month", "Once  a month", "Every  other week", "Once  a week", "Every  other day", "Daily"),labels=c("1","2","3","4","5","6","7","8")) 
  df$G04Q16_SQ002 <- factor(as.factor(df$G04Q16_SQ002),levels=c("Never", "Very rarely", "Every  other month", "Once  a month", "Every  other week", "Once  a week", "Every  other day", "Daily"),labels=c("1","2","3","4","5","6","7","8")) 
  df$G04Q16_SQ003 <- factor(as.factor(df$G04Q16_SQ003),levels=c("Never", "Very rarely", "Every  other month", "Once  a month", "Every  other week", "Once  a week", "Every  other day", "Daily"),labels=c("1","2","3","4","5","6","7","8")) 
  df$G04Q16_SQ004 <- factor(as.factor(df$G04Q16_SQ004),levels=c("Never", "Very rarely", "Every  other month", "Once  a month", "Every  other week", "Once  a week", "Every  other day", "Daily"),labels=c("1","2","3","4","5","6","7","8")) 
  df$G03Q15_SQ001 <- factor(as.factor(df$G03Q15_SQ001),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7")) 
  df$G03Q15_SQ002 <- factor(as.factor(df$G03Q15_SQ002),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7")) 
  df$G03Q15_SQ003 <- factor(as.factor(df$G03Q15_SQ003),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7")) 
  df$G03Q15_SQ004 <- factor(as.factor(df$G03Q15_SQ004),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7")) 
  df$G03Q15_SQ005 <- factor(as.factor(df$G03Q15_SQ005),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7"))
  df$G03Q15_SQ006 <- factor(as.factor(df$G03Q15_SQ006),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7"))
  df$G03Q15_SQ007 <- factor(as.factor(df$G03Q15_SQ007),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7"))
  df$G03Q15_SQ008 <- factor(as.factor(df$G03Q15_SQ008),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7"))
  df$G03Q15_SQ009 <- factor(as.factor(df$G03Q15_SQ009),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7")) 
  df$G03Q15_SQ010 <- factor(as.factor(df$G03Q15_SQ010),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7")) 
  df$G03Q15_SQ011 <- factor(as.factor(df$G03Q15_SQ011),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7")) 
  df$G03Q15_SQ012 <- factor(as.factor(df$G03Q15_SQ012),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7")) 
  df$G03Q15_SQ013 <- factor(as.factor(df$G03Q15_SQ013),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7")) 
  df$G03Q15_SQ014 <- factor(as.factor(df$G03Q15_SQ014),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7")) 
  df$G03Q15_SQ015 <- factor(as.factor(df$G03Q15_SQ015),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7"))
  df$G03Q15_SQ016 <- factor(as.factor(df$G03Q15_SQ016),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7")) 
  df$G03Q15_SQ017 <- factor(as.factor(df$G03Q15_SQ017),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7")) 
  df$G03Q15_SQ018 <- factor(as.factor(df$G03Q15_SQ018),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7")) 
  df$G03Q15_SQ019 <- factor(as.factor(df$G03Q15_SQ019),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7")) 
  df$G03Q15_SQ020 <- factor(as.factor(df$G03Q15_SQ020),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7")) 
  df$G03Q15_SQ021 <- factor(as.factor(df$G03Q15_SQ021),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7")) 
  df$G03Q15_SQ022 <-factor(as.factor(df$G03Q15_SQ022),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7"))
  df$G03Q15_SQ023 <-factor(as.factor(df$G03Q15_SQ023),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7")) 
  df$G03Q15_SQ024 <- factor(as.factor(df$G03Q15_SQ024),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7"))
  df$G03Q15_SQ025 <- factor(as.factor(df$G03Q15_SQ025),levels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"),labels=c("1","2","3","4","5","6","7")) 
  
  df <- rename(df, AT1=G02Q09_SQ001)  
  df <- rename(df, AT2=G02Q10_SQ001) 
  df <- rename(df, AT3=G02Q11_SQ001)
  df <- rename(df, AT4=G02Q12_SQ001)
  df <- rename(df, AT5=G02Q13_SQ001)
  df <- rename(df, AT6=G02Q14_SQ001)
  df <- rename(df, AT7=G02Q14_SQ002)
  df <- rename(df, SN1=G03Q15_SQ001)
  df <- rename(df, SN2=G03Q15_SQ002)
  df <- rename(df, SN3=G03Q15_SQ003)
  df <- rename(df, SN4=G03Q15_SQ004)
  df <- rename(df, SN5=G03Q15_SQ005)
  df <- rename(df, PBC1=G03Q15_SQ006)
  df <- rename(df, PBC2=G03Q15_SQ007)
  df <- rename(df, PBC3=G03Q15_SQ008)
  df <- rename(df, PBC4=G03Q15_SQ009)
  df <- rename(df, PBC5=G03Q15_SQ010)
  df <- rename(df, PBC6=G03Q15_SQ011)
  df <- rename(df, AC1=G03Q15_SQ012)
  df <- rename(df, AC2=G03Q15_SQ013)
  df <- rename(df, AC3=G03Q15_SQ014)
  df <- rename(df, AR1=G03Q15_SQ015)
  df <- rename(df, AR2=G03Q15_SQ016)
  df <- rename(df, AR3=G03Q15_SQ017)
  df <- rename(df, PN1=G03Q15_SQ018)
  df <- rename(df, PN2=G03Q15_SQ019)
  df <- rename(df, PN3=G03Q15_SQ020)
  df <- rename(df, BI1=G03Q15_SQ021)
  df <- rename(df, BI2=G03Q15_SQ022)
  df <- rename(df, BI3=G03Q15_SQ023)
  df <- rename(df, BI4=G03Q15_SQ024)
  df <- rename(df, BI5=G03Q15_SQ025)
  df <- rename(df, PB1=G04Q16_SQ001)
  df <- rename(df, PB2=G04Q16_SQ002)
  df <- rename(df, PB3=G04Q16_SQ003)
  df <- rename(df, PB4=G04Q16_SQ004)
}
#-----
syntax2 <- function(df){
df$G02Q09_SQ001 <- fct_recode(df$G02Q09_SQ001,"1"="Very unpleasant", "2"="Unpleasant", "3"="Somewhat unpleasant", "4"="Neutral", "5"="Somewhat pleasant", "6"="Pleasant", "7"="Very pleasant") 
df$G02Q10_SQ001 <- fct_recode(df$G02Q10_SQ001,"1"="Very inconvenient", "2"="Inconvenient", "3"="Somewhat inconvenient", "4"="Neutral", "5"="Somewhat  convenient", "6"="Convenient", "7"="Very  convenient") 
df$G02Q11_SQ001 <- fct_recode(df$G02Q11_SQ001,"1"="Very unsafe", "2"="Unsafe", "3"="Somewhat unsafe", "4"="Neutral", "5"="Somewhat safe", "6"="Safe", "7"="Very safe") 
df$G02Q12_SQ001 <- fct_recode(df$G02Q12_SQ001,"1"="Very unimportant", "2"="Unimportant", "3"="Somewhat unimportant", "4"="Neutral", "5"="Somewhat important", "6"="Important", "7"="Very important") 
df$G02Q13_SQ001 <- fct_recode(df$G02Q13_SQ001,"1"="Very unhelpful", "2"="Unhelpful", "3"="Somewhat unhelpful", "4"="Neutral", "5"="Somewhat helpful", "6"="Helpful", "7"="Very helpful") 
df$G02Q14_SQ001 <- fct_recode(df$G02Q14_SQ001,"1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neutral", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G02Q14_SQ002 <- fct_recode(df$G02Q14_SQ002,"1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neutral", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G04Q16_SQ001 <- fct_recode(df$G04Q16_SQ001,"1"="Never", "2"="Very rarely", "3"="Every other month", "4"="Once  a month", "5"="Every  other week", "6"="Once a week", "7"="Every other day", "8"="Daily") 
df$G04Q16_SQ002 <- fct_recode(df$G04Q16_SQ002,"1"="Never", "2"="Very rarely", "3"="Every other month", "4"="Once  a month", "5"="Every  other week", "6"="Once a week", "7"="Every other day", "8"="Daily") 
df$G04Q16_SQ003 <- fct_recode(df$G04Q16_SQ003,"1"="Never", "2"="Very rarely", "3"="Every other month", "4"="Once  a month", "5"="Every  other week", "6"="Once a week", "7"="Every other day", "8"="Daily") 
df$G04Q16_SQ004 <- fct_recode(df$G04Q16_SQ004,"1"="Never", "2"="Very rarely", "3"="Every other month", "4"="Once  a month", "5"="Every  other week", "6"="Once a week", "7"="Every other day", "8"="Daily") 
df$G03Q15_SQ001 <- fct_recode(df$G03Q15_SQ001, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G03Q15_SQ002 <- fct_recode(df$G03Q15_SQ002, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G03Q15_SQ003 <- fct_recode(df$G03Q15_SQ003, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G03Q15_SQ004 <- fct_recode(df$G03Q15_SQ004, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G03Q15_SQ005 <- fct_recode(df$G03Q15_SQ005, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree")
df$G03Q15_SQ006 <- fct_recode(df$G03Q15_SQ006, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree")
df$G03Q15_SQ007 <- fct_recode(df$G03Q15_SQ007, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree")
df$G03Q15_SQ008 <- fct_recode(df$G03Q15_SQ008, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G03Q15_SQ009 <- fct_recode(df$G03Q15_SQ009, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G03Q15_SQ010 <- fct_recode(df$G03Q15_SQ010, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G03Q15_SQ011 <- fct_recode(df$G03Q15_SQ011, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G03Q15_SQ012 <- fct_recode(df$G03Q15_SQ012, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G03Q15_SQ013 <- fct_recode(df$G03Q15_SQ013, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G03Q15_SQ014 <- fct_recode(df$G03Q15_SQ014, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G03Q15_SQ015 <- fct_recode(df$G03Q15_SQ015, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G03Q15_SQ016 <- fct_recode(df$G03Q15_SQ016, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G03Q15_SQ017 <- fct_recode(df$G03Q15_SQ017, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G03Q15_SQ018 <- fct_recode(df$G03Q15_SQ018, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G03Q15_SQ019 <- fct_recode(df$G03Q15_SQ019, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G03Q15_SQ020 <- fct_recode(df$G03Q15_SQ020, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G03Q15_SQ021 <- fct_recode(df$G03Q15_SQ021, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G03Q15_SQ022 <-fct_recode(df$G03Q15_SQ022, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G03Q15_SQ023 <-fct_recode(df$G03Q15_SQ023, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G03Q15_SQ024 <- fct_recode(df$G03Q15_SQ024, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 
df$G03Q15_SQ025 <- fct_recode(df$G03Q15_SQ025, "1"="Strongly  disagree", "2"="Disagree", "3"="Somewhat  disagree", "4"="Neither agree  nor disagree", "5"="Somewhat  agree", "6"="Agree", "7"="Strongly agree") 

df <- rename(df, AT1=G02Q09_SQ001)  
df <- rename(df, AT2=G02Q10_SQ001) 
df <- rename(df, AT3=G02Q11_SQ001)
df <- rename(df, AT4=G02Q12_SQ001)
df <- rename(df, AT5=G02Q13_SQ001)
df <- rename(df, AT6=G02Q14_SQ001)
df <- rename(df, AT7=G02Q14_SQ002)
df <- rename(df, SN1=G03Q15_SQ001)
df <- rename(df, SN2=G03Q15_SQ002)
df <- rename(df, SN3=G03Q15_SQ003)
df <- rename(df, SN4=G03Q15_SQ004)
df <- rename(df, SN5=G03Q15_SQ005)
df <- rename(df, PBC1=G03Q15_SQ006)
df <- rename(df, PBC2=G03Q15_SQ007)
df <- rename(df, PBC3=G03Q15_SQ008)
df <- rename(df, PBC4=G03Q15_SQ009)
df <- rename(df, PBC5=G03Q15_SQ010)
df <- rename(df, PBC6=G03Q15_SQ011)
df <- rename(df, AC1=G03Q15_SQ012)
df <- rename(df, AC2=G03Q15_SQ013)
df <- rename(df, AC3=G03Q15_SQ014)
df <- rename(df, AR1=G03Q15_SQ015)
df <- rename(df, AR2=G03Q15_SQ016)
df <- rename(df, AR3=G03Q15_SQ017)
df <- rename(df, PN1=G03Q15_SQ018)
df <- rename(df, PN2=G03Q15_SQ019)
df <- rename(df, PN3=G03Q15_SQ020)
df <- rename(df, BI1=G03Q15_SQ021)
df <- rename(df, BI2=G03Q15_SQ022)
df <- rename(df, BI3=G03Q15_SQ023)
df <- rename(df, BI4=G03Q15_SQ024)
df <- rename(df, BI5=G03Q15_SQ025)
df <- rename(df, PB1=G04Q16_SQ001)
df <- rename(df, PB2=G04Q16_SQ002)
df <- rename(df, PB3=G04Q16_SQ003)
df <- rename(df, PB4=G04Q16_SQ004)
}
