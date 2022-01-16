
# LimeSurvey Field type: F
data[, 1] <- as.numeric(data[, 1])
attributes(data)$variable.labels[1] <- "id"
names(data)[1] <- "id"
# LimeSurvey Field type: DATETIME23.2
data[, 2] <- as.character(data[, 2])
attributes(data)$variable.labels[2] <- "submitdate"
names(data)[2] <- "submitdate"
# LimeSurvey Field type: F
data[, 3] <- as.numeric(data[, 3])
attributes(data)$variable.labels[3] <- "lastpage"
names(data)[3] <- "lastpage"
# LimeSurvey Field type: A
data[, 4] <- as.character(data[, 4])
attributes(data)$variable.labels[4] <- "startlanguage"
names(data)[4] <- "startlanguage"
# LimeSurvey Field type: A
data[, 5] <- as.character(data[, 5])
attributes(data)$variable.labels[5] <- "seed"
names(data)[5] <- "seed"
# LimeSurvey Field type: DATETIME23.2
data[, 6] <- as.character(data[, 6])
attributes(data)$variable.labels[6] <- "startdate"
names(data)[6] <- "startdate"
# LimeSurvey Field type: DATETIME23.2
data[, 7] <- as.character(data[, 7])
attributes(data)$variable.labels[7] <- "datestamp"
names(data)[7] <- "datestamp"
# LimeSurvey Field type: A
data[, 8] <- as.character(data[, 8])
attributes(data)$variable.labels[8] <- "refurl"
names(data)[8] <- "refurl"
# LimeSurvey Field type: A
data[, 9] <- as.character(data[, 9])
attributes(data)$variable.labels[9] <- "Are you currently living in a home that is equipped with a smart home system?"
#data[, 9] <- factor(data[, 9], levels=c("AO01","AO02"),labels=c("Yes", "No"))
names(data)[9] <- "G01Q01"
# LimeSurvey Field type: A
data[, 10] <- as.character(data[, 10])
attributes(data)$variable.labels[10] <- "How long have you been living in a home with a smart home system?"
#data[, 10] <- factor(data[, 10], levels=c("AO01","AO02","AO03","AO04","AO05","AO06"),labels=c("less than six months", "6-12 months", "1-3 years", "4-6 years", "7-10 years", "more than 10 years"))
names(data)[10] <- "G01Q02"
# LimeSurvey Field type: A
data[, 11] <- as.character(data[, 11])
attributes(data)$variable.labels[11] <- "What is the name of the smart home system that you are using?"
#data[, 11] <- factor(data[, 11], levels=c("AO01","AO03","AO02"),labels=c("Home Assistant", "Vera", "OpenHAB"))
names(data)[11] <- "G01Q03"
# LimeSurvey Field type: A
data[, 12] <- as.character(data[, 12])
attributes(data)$variable.labels[12] <- "[Other] What is the name of the smart home system that you are using?"
names(data)[12] <- "G01Q03_other"
# LimeSurvey Field type: A
data[, 13] <- as.character(data[, 13])
attributes(data)$variable.labels[13] <- "A smart home energy management system is a system that provides services to monitor and manage electricity generation, storage, and/or consumption in a home, and includes some form of automated planning. (Homan, 2020; Zhou et al., 2016)Are you currently using an energy management system as part of your smart home?"
#data[, 13] <- factor(data[, 13], levels=c("AO01","AO02","AO03"),labels=c("Yes", "No", "I don\'t know."))
names(data)[13] <- "G01Q04"
# LimeSurvey Field type: F
#data[, 14] <- as.numeric(data[, 14])
attributes(data)$variable.labels[14] <- "[Smart thermostat] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 14] <- factor(data[, 14], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[14] <- "G01Q05_SQ001"
# LimeSurvey Field type: F
#data[, 15] <- as.numeric(data[, 15])
attributes(data)$variable.labels[15] <- "[Smart plugs] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 15] <- factor(data[, 15], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[15] <- "G01Q05_SQ002"
# LimeSurvey Field type: F
#data[, 16] <- as.numeric(data[, 16])
attributes(data)$variable.labels[16] <- "[Smart meter] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 16] <- factor(data[, 16], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[16] <- "G01Q05_SQ003"
# LimeSurvey Field type: F
#[, 17] <- as.numeric(data[, 17])
attributes(data)$variable.labels[17] <- "[Heating system] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 17] <- factor(data[, 17], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[17] <- "G01Q05_SQ004"
# LimeSurvey Field type: F
#data[, 18] <- as.numeric(data[, 18])
attributes(data)$variable.labels[18] <- "[Solar panels (or other energy-generating devices)] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 18] <- factor(data[, 18], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[18] <- "G01Q05_SQ005"
# LimeSurvey Field type: F
#data[, 19] <- as.numeric(data[, 19])
attributes(data)$variable.labels[19] <- "[Energy storage system] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 19] <- factor(data[, 19], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[19] <- "G01Q05_SQ013"
# LimeSurvey Field type: F
#data[, 20] <- as.numeric(data[, 20])
attributes(data)$variable.labels[20] <- "[Light sources (for example smart bulbs)] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 20] <- factor(data[, 20], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[20] <- "G01Q05_SQ006"
# LimeSurvey Field type: F
#data[, 21] <- as.numeric(data[, 21])
attributes(data)$variable.labels[21] <- "[Air conditioning] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 21] <- factor(data[, 21], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[21] <- "G01Q05_SQ007"
# LimeSurvey Field type: F
#data[, 22] <- as.numeric(data[, 22])
attributes(data)$variable.labels[22] <- "[Ventilation] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 22] <- factor(data[, 22], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[22] <- "G01Q05_SQ008"
# LimeSurvey Field type: F
#data[, 23] <- as.numeric(data[, 23])
attributes(data)$variable.labels[23] <- "[Water heater] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 23] <- factor(data[, 23], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[23] <- "G01Q05_SQ009"
# LimeSurvey Field type: F
#data[, 24] <- as.numeric(data[, 24])
attributes(data)$variable.labels[24] <- "[Shading devices] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 24] <- factor(data[, 24], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[24] <- "G01Q05_SQ010"
# LimeSurvey Field type: F
#data[, 25] <- as.numeric(data[, 25])
attributes(data)$variable.labels[25] <- "[Switches] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 25] <- factor(data[, 25], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[25] <- "G01Q05_SQ011"
# LimeSurvey Field type: F
#data[, 26] <- as.numeric(data[, 26])
attributes(data)$variable.labels[26] <- "[Garage door controls] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 26] <- factor(data[, 26], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[26] <- "G01Q05_SQ014"
# LimeSurvey Field type: F
#data[, 27] <- as.numeric(data[, 27])
attributes(data)$variable.labels[27] <- "[Door locks] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 27] <- factor(data[, 27], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[27] <- "G01Q05_SQ015"
# LimeSurvey Field type: F
#data[, 28] <- as.numeric(data[, 28])
attributes(data)$variable.labels[28] <- "[Motion sensors] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 28] <- factor(data[, 28], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[28] <- "G01Q05_SQ016"
# LimeSurvey Field type: F
#data[, 29] <- as.numeric(data[, 29])
attributes(data)$variable.labels[29] <- "[Door and window sensors] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 29] <- factor(data[, 29], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[29] <- "G01Q05_SQ017"
# LimeSurvey Field type: F
#data[, 30] <- as.numeric(data[, 30])
attributes(data)$variable.labels[30] <- "[Environmental sensors (such as sensors for temperature, humidity, light)] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 30] <- factor(data[, 30], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[30] <- "G01Q05_SQ034"
# LimeSurvey Field type: F
#data[, 31] <- as.numeric(data[, 31])
attributes(data)$variable.labels[31] <- "[Flood sensors] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 31] <- factor(data[, 31], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[31] <- "G01Q05_SQ035"
# LimeSurvey Field type: F
#data[, 32] <- as.numeric(data[, 32])
attributes(data)$variable.labels[32] <- "[Fire, smoke, or gas detection] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 32] <- factor(data[, 32], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[32] <- "G01Q05_SQ029"
# LimeSurvey Field type: F
#data[, 33] <- as.numeric(data[, 33])
attributes(data)$variable.labels[33] <- "[Security cameras] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 33] <- factor(data[, 33], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[33] <- "G01Q05_SQ031"
# LimeSurvey Field type: F
#data[, 34] <- as.numeric(data[, 34])
attributes(data)$variable.labels[34] <- "[Refrigerator] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 34] <- factor(data[, 34], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[34] <- "G01Q05_SQ019"
# LimeSurvey Field type: F
#data[, 35] <- as.numeric(data[, 35])
attributes(data)$variable.labels[35] <- "[Stove and/or oven] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 35] <- factor(data[, 35], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[35] <- "G01Q05_SQ021"
# LimeSurvey Field type: F
#data[, 36] <- as.numeric(data[, 36])
attributes(data)$variable.labels[36] <- "[Dishwasher] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 36] <- factor(data[, 36], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[36] <- "G01Q05_SQ022"
# LimeSurvey Field type: F
#data[, 37] <- as.numeric(data[, 37])
attributes(data)$variable.labels[37] <- "[Washing machine] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 37] <- factor(data[, 37], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[37] <- "G01Q05_SQ023"
# LimeSurvey Field type: F
#data[, 38] <- as.numeric(data[, 38])
attributes(data)$variable.labels[38] <- "[Dryer] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 38] <- factor(data[, 38], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[38] <- "G01Q05_SQ024"
# LimeSurvey Field type: F
#data[, 39] <- as.numeric(data[, 39])
attributes(data)$variable.labels[39] <- "[Coffee machine] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 39] <- factor(data[, 39], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[39] <- "G01Q05_SQ025"
# LimeSurvey Field type: F
#data[, 40] <- as.numeric(data[, 40])
attributes(data)$variable.labels[40] <- "[Microwave] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 40] <- factor(data[, 40], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[40] <- "G01Q05_SQ026"
# LimeSurvey Field type: F
#data[, 41] <- as.numeric(data[, 41])
attributes(data)$variable.labels[41] <- "[Vacuum cleaner] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 41] <- factor(data[, 41], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[41] <- "G01Q05_SQ027"
# LimeSurvey Field type: F
#data[, 42] <- as.numeric(data[, 42])
attributes(data)$variable.labels[42] <- "[Mowing robot] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 42] <- factor(data[, 42], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[42] <- "G01Q05_SQ028"
# LimeSurvey Field type: F
#data[, 43] <- as.numeric(data[, 43])
attributes(data)$variable.labels[43] <- "[Smart TV] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 43] <- factor(data[, 43], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[43] <- "G01Q05_SQ018"
# LimeSurvey Field type: F
#data[, 44] <- as.numeric(data[, 44])
attributes(data)$variable.labels[44] <- "[Sound system] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 44] <- factor(data[, 44], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[44] <- "G01Q05_SQ012"
# LimeSurvey Field type: F
#data[, 45] <- as.numeric(data[, 45])
attributes(data)$variable.labels[45] <- "[Smart speaker] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 45] <- factor(data[, 45], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[45] <- "G01Q05_SQ032"
# LimeSurvey Field type: F
#data[, 46] <- as.numeric(data[, 46])
attributes(data)$variable.labels[46] <- "[Streaming devices (for example Amazon Fire TV stick, Google Chromecast)] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 46] <- factor(data[, 46], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[46] <- "G01Q05_SQ033"
# LimeSurvey Field type: F
#data[, 47] <- as.numeric(data[, 47])
attributes(data)$variable.labels[47] <- "[Health-related devices] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 47] <- factor(data[, 47], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[47] <- "G01Q05_SQ030"
# LimeSurvey Field type: F
#data[, 48] <- as.numeric(data[, 48])
attributes(data)$variable.labels[48] <- "[Electric vehicle] Which products or systems are currently connected to your smart home? Please tick all that apply."
#data[, 48] <- factor(data[, 48], levels=c(1,0),labels=c("Yes", "Not selected"))
names(data)[48] <- "G01Q05_SQ020"
# LimeSurvey Field type: A
data[, 49] <- as.character(data[, 49])
attributes(data)$variable.labels[49] <- "[Other] Which products or systems are currently connected to your smart home? Please tick all that apply."
names(data)[49] <- "G01Q05_other"
# LimeSurvey Field type: A
data[, 50] <- as.character(data[, 50])
attributes(data)$variable.labels[50] <- "Which person in your household is mainly responsible for your smart home system?"
#data[, 50] <- factor(data[, 50], levels=c("AO01","AO02","AO03"),labels=c("I am.", "Responsibility is equally distributed among household members.", "Another household member."))
names(data)[50] <- "G01Q06"
# LimeSurvey Field type: A
data[, 51] <- as.character(data[, 51])
attributes(data)$variable.labels[51] <- "[Other] Which person in your household is mainly responsible for your smart home system?"
names(data)[51] <- "G01Q06_other"
# LimeSurvey Field type: A
data[, 52] <- as.character(data[, 52])
attributes(data)$variable.labels[52] <- "[Interest in smart home technology] [Importance] Why did you decide to install a smart home system in your home? Please allocate points (0-100) to the listed options depending on their importance to you. All points together must equal 100. Only integer values may be entered in these fields."
names(data)[52] <- "G01Q07_SQ001_SQ011"
# LimeSurvey Field type: A
data[, 53] <- as.character(data[, 53])
attributes(data)$variable.labels[53] <- "[To be more sustainable and save energy] [Importance] Why did you decide to install a smart home system in your home? Please allocate points (0-100) to the listed options depending on their importance to you. All points together must equal 100. Only integer values may be entered in these fields."
names(data)[53] <- "G01Q07_SQ002_SQ011"
# LimeSurvey Field type: A
data[, 54] <- as.character(data[, 54])
attributes(data)$variable.labels[54] <- "[For fun and entertainment purposes] [Importance] Why did you decide to install a smart home system in your home? Please allocate points (0-100) to the listed options depending on their importance to you. All points together must equal 100. Only integer values may be entered in these fields."
names(data)[54] <- "G01Q07_SQ003_SQ011"
# LimeSurvey Field type: A
data[, 55] <- as.character(data[, 55])
attributes(data)$variable.labels[55] <- "[To increase comfort and convenience at home] [Importance] Why did you decide to install a smart home system in your home? Please allocate points (0-100) to the listed options depending on their importance to you. All points together must equal 100. Only integer values may be entered in these fields."
names(data)[55] <- "G01Q07_SQ004_SQ011"
# LimeSurvey Field type: A
data[, 56] <- as.character(data[, 56])
attributes(data)$variable.labels[56] <- "[To save money] [Importance] Why did you decide to install a smart home system in your home? Please allocate points (0-100) to the listed options depending on their importance to you. All points together must equal 100. Only integer values may be entered in these fields."
names(data)[56] <- "G01Q07_SQ005_SQ011"
# LimeSurvey Field type: A
data[, 57] <- as.character(data[, 57])
attributes(data)$variable.labels[57] <- "[To better care for elders or people with disabilities (Ambient Assisted Living)] [Importance] Why did you decide to install a smart home system in your home? Please allocate points (0-100) to the listed options depending on their importance to you. All points together must equal 100. Only integer values may be entered in these fields."
names(data)[57] <- "G01Q07_SQ006_SQ011"
# LimeSurvey Field type: A
data[, 58] <- as.character(data[, 58])
attributes(data)$variable.labels[58] <- "[To increase safety and security at home] [Importance] Why did you decide to install a smart home system in your home? Please allocate points (0-100) to the listed options depending on their importance to you. All points together must equal 100. Only integer values may be entered in these fields."
names(data)[58] <- "G01Q07_SQ007_SQ011"
# LimeSurvey Field type: A
data[, 59] <- as.character(data[, 59])
attributes(data)$variable.labels[59] <- "[To have more control over my home] [Importance] Why did you decide to install a smart home system in your home? Please allocate points (0-100) to the listed options depending on their importance to you. All points together must equal 100. Only integer values may be entered in these fields."
names(data)[59] <- "G01Q07_SQ008_SQ011"
# LimeSurvey Field type: A
data[, 60] <- as.character(data[, 60])
attributes(data)$variable.labels[60] <- "[It was already installed in the home.] [Importance] Why did you decide to install a smart home system in your home? Please allocate points (0-100) to the listed options depending on their importance to you. All points together must equal 100. Only integer values may be entered in these fields."
names(data)[60] <- "G01Q07_SQ009_SQ011"
# LimeSurvey Field type: A
data[, 61] <- as.character(data[, 61])
attributes(data)$variable.labels[61] <- "[Other] [Importance] Why did you decide to install a smart home system in your home? Please allocate points (0-100) to the listed options depending on their importance to you. All points together must equal 100. Only integer values may be entered in these fields."
names(data)[61] <- "G01Q07_SQ010_SQ011"
# LimeSurvey Field type: A
data[, 62] <- as.character(data[, 62])
attributes(data)$variable.labels[62] <- "From the previous question: What other reasons influenced your decision to install a smart home system?"
names(data)[62] <- "G01Q08"
# LimeSurvey Field type: A
data[, 63] <- as.character(data[, 63])
attributes(data)$variable.labels[63] <- "[I think that using my smart home for energy management is/would be …] Please indicate how you would rate the following statement."
#data[, 63] <- factor(data[, 63], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Very  unpleasant", "Unpleasant", "Somewhat  unpleasant", "Neutral", "Somewhat  pleasant", "Pleasant", "Very  pleasant"))
names(data)[63] <- "G02Q09_SQ001"
# LimeSurvey Field type: A
data[, 64] <- as.character(data[, 64])
attributes(data)$variable.labels[64] <- "[I think that using my smart home for energy management is/would be …] Please indicate how you would rate the following statement."
#data[, 64] <- factor(data[, 64], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Very  inconvenient", "Inconvenient", "Somewhat  inconvenient", "Neutral", "Somewhat  convenient", "Convenient", "Very  convenient"))
names(data)[64] <- "G02Q10_SQ001"
# LimeSurvey Field type: A
data[, 65] <- as.character(data[, 65])
attributes(data)$variable.labels[65] <- "[I think that using my smart home for energy management is/would be ….] Please indicate how you would rate the following statement."
#data[, 65] <- factor(data[, 65], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Very  unsafe", "Unsafe", "Somewhat  unsafe", "Neutral", "Somewhat  safe", "Safe", "Very  safe"))
names(data)[65] <- "G02Q11_SQ001"
# LimeSurvey Field type: A
data[, 66] <- as.character(data[, 66])
attributes(data)$variable.labels[66] <- "[I think that using my smart home to reduce my energy consumption is …] Please indicate how you would rate the following statement."
#data[, 66] <- factor(data[, 66], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Very  unimportant", "Unimportant", "Somewhat  unimportant", "Neutral", "Somewhat  important", "Important", "Very  important"))
names(data)[66] <- "G02Q12_SQ001"
# LimeSurvey Field type: A
data[, 67] <- as.character(data[, 67])
attributes(data)$variable.labels[67] <- "[I think that using my smart home is/would be …. to reduce my energy consumption.] Please indicate how you would rate the following statement."
#data[, 67] <- factor(data[, 67], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Very  unhelpful", "Unhelpful", "Somewhat  unhelpful", "Neutral", "Somewhat  helpful", "Helpful", "Very  helpful"))
names(data)[67] <- "G02Q13_SQ001"
# LimeSurvey Field type: A
data[, 68] <- as.character(data[, 68])
attributes(data)$variable.labels[68] <- "[I think that using my smart home helps/would help me to better understand my own energy consumption.] Please indicate how you would rate the following statements."
#data[, 68] <- factor(data[, 68], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neutral", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[68] <- "G02Q14_SQ001"
# LimeSurvey Field type: A
data[, 69] <- as.character(data[, 69])
attributes(data)$variable.labels[69] <- "[My smart home helps/would help me to save money by reducing my energy consumption.] Please indicate how you would rate the following statements."
#data[, 69] <- factor(data[, 69], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neutral", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[69] <- "G02Q14_SQ002"
# LimeSurvey Field type: A
data[, 70] <- as.character(data[, 70])
attributes(data)$variable.labels[70] <- "[Environmental sustainability and climate change are important topics for people that are important to me. ] Please indicate how much you agree or disagree with the following statements."
#data[, 70] <- factor(data[, 70], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[70] <- "G03Q15_SQ001"
# LimeSurvey Field type: A
data[, 71] <- as.character(data[, 71])
attributes(data)$variable.labels[71] <- "[People who are important to me are trying to reduce their energy consumption. ] Please indicate how much you agree or disagree with the following statements."
#data[, 71] <- factor(data[, 71], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[71] <- "G03Q15_SQ002"
# LimeSurvey Field type: A
data[, 72] <- as.character(data[, 72])
attributes(data)$variable.labels[72] <- "[People who are important to me would approve of me trying to reduce my energy consumption.] Please indicate how much you agree or disagree with the following statements."
#data[, 72] <- factor(data[, 72], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[72] <- "G03Q15_SQ003"
# LimeSurvey Field type: A
data[, 73] <- as.character(data[, 73])
attributes(data)$variable.labels[73] <- "[People who are important to me are trying to reduce their energy consumption by using their smart home.] Please indicate how much you agree or disagree with the following statements."
#data[, 73] <- factor(data[, 73], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[73] <- "G03Q15_SQ004"
# LimeSurvey Field type: A
data[, 74] <- as.character(data[, 74])
attributes(data)$variable.labels[74] <- "[People who are important to me think that I should reduce my energy consumption.] Please indicate how much you agree or disagree with the following statements."
#data[, 74] <- factor(data[, 74], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[74] <- "G03Q15_SQ005"
# LimeSurvey Field type: A
data[, 75] <- as.character(data[, 75])
attributes(data)$variable.labels[75] <- "[I know how to save energy with my smart home.] Please indicate how much you agree or disagree with the following statements."
#data[, 75] <- factor(data[, 75], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[75] <- "G03Q15_SQ006"
# LimeSurvey Field type: A
data[, 76] <- as.character(data[, 76])
attributes(data)$variable.labels[76] <- "[Controlling my smart home through the user interface is easy.] Please indicate how much you agree or disagree with the following statements."
#data[, 76] <- factor(data[, 76], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[76] <- "G03Q15_SQ007"
# LimeSurvey Field type: A
data[, 77] <- as.character(data[, 77])
attributes(data)$variable.labels[77] <- "[I have enough time to set up and adjust my smart home to reduce my energy consumption.] Please indicate how much you agree or disagree with the following statements."
#data[, 77] <- factor(data[, 77], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[77] <- "G03Q15_SQ008"
# LimeSurvey Field type: A
data[, 78] <- as.character(data[, 78])
attributes(data)$variable.labels[78] <- "[I have enough skills and experience to reduce my energy consumption with my smart home.] Please indicate how much you agree or disagree with the following statements."
#data[, 78] <- factor(data[, 78], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[78] <- "G03Q15_SQ009"
# LimeSurvey Field type: A
data[, 79] <- as.character(data[, 79])
attributes(data)$variable.labels[79] <- "[Reducing my energy consumption is out of my personal control.] Please indicate how much you agree or disagree with the following statements."
#data[, 79] <- factor(data[, 79], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[79] <- "G03Q15_SQ010"
# LimeSurvey Field type: A
data[, 80] <- as.character(data[, 80])
attributes(data)$variable.labels[80] <- "[Learning how to use my smart home for energy management is simple.] Please indicate how much you agree or disagree with the following statements."
#data[, 80] <- factor(data[, 80], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[80] <- "G03Q15_SQ011"
# LimeSurvey Field type: A
data[, 81] <- as.character(data[, 81])
attributes(data)$variable.labels[81] <- "[My energy consumption at home influences the environment.] Please indicate how much you agree or disagree with the following statements."
#data[, 81] <- factor(data[, 81], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[81] <- "G03Q15_SQ012"
# LimeSurvey Field type: A
data[, 82] <- as.character(data[, 82])
attributes(data)$variable.labels[82] <- "[I am aware of the importance to reduce energy consumption for environmental reasons.] Please indicate how much you agree or disagree with the following statements."
#data[, 82] <- factor(data[, 82], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[82] <- "G03Q15_SQ013"
# LimeSurvey Field type: A
data[, 83] <- as.character(data[, 83])
attributes(data)$variable.labels[83] <- "[I am concerned about climate change and its consequences.] Please indicate how much you agree or disagree with the following statements."
#data[, 83] <- factor(data[, 83], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[83] <- "G03Q15_SQ014"
# LimeSurvey Field type: A
data[, 84] <- as.character(data[, 84])
attributes(data)$variable.labels[84] <- "[Every individual is responsible to be more mindful about their energy consumption.] Please indicate how much you agree or disagree with the following statements."
#data[, 84] <- factor(data[, 84], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[84] <- "G03Q15_SQ015"
# LimeSurvey Field type: A
data[, 85] <- as.character(data[, 85])
attributes(data)$variable.labels[85] <- "[I feel personally obliged to reduce my energy consumption to the best of my ability, even if it is a small act of my own.] Please indicate how much you agree or disagree with the following statements."
#data[, 85] <- factor(data[, 85], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[85] <- "G03Q15_SQ016"
# LimeSurvey Field type: A
data[, 86] <- as.character(data[, 86])
attributes(data)$variable.labels[86] <- "[I feel guilty if I unnecessarily increase my energy consumption to improve my own comfort and convenience.] Please indicate how much you agree or disagree with the following statements."
#data[, 86] <- factor(data[, 86], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[86] <- "G03Q15_SQ017"
# LimeSurvey Field type: A
data[, 87] <- as.character(data[, 87])
attributes(data)$variable.labels[87] <- "[Environmental sustainability is important to me.] Please indicate how much you agree or disagree with the following statements."
#data[, 87] <- factor(data[, 87], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[87] <- "G03Q15_SQ018"
# LimeSurvey Field type: A
data[, 88] <- as.character(data[, 88])
attributes(data)$variable.labels[88] <- "[It is important to me to focus on reducing my energy consumption instead of using my smart home for entertainment purposes.] Please indicate how much you agree or disagree with the following statements."
#data[, 88] <- factor(data[, 88], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[88] <- "G03Q15_SQ019"
# LimeSurvey Field type: A
data[, 89] <- as.character(data[, 89])
attributes(data)$variable.labels[89] <- "[I feel a moral obligation to reduce my energy consumption with the help of my smart home system.] Please indicate how much you agree or disagree with the following statements."
#data[, 89] <- factor(data[, 89], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[89] <- "G03Q15_SQ020"
# LimeSurvey Field type: A
data[, 90] <- as.character(data[, 90])
attributes(data)$variable.labels[90] <- "[I want to continue or start to use my smart home to manage my energy consumption in the foreseeable future.] Please indicate how much you agree or disagree with the following statements."
#data[, 90] <- factor(data[, 90], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[90] <- "G03Q15_SQ021"
# LimeSurvey Field type: A
data[, 91] <- as.character(data[, 91])
attributes(data)$variable.labels[91] <- "[I want to (continue to) reduce my energy consumption with the help of my smart home in the foreseeable future.] Please indicate how much you agree or disagree with the following statements."
#data[, 91] <- factor(data[, 91], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[91] <- "G03Q15_SQ022"
# LimeSurvey Field type: A
data[, 92] <- as.character(data[, 92])
attributes(data)$variable.labels[92] <- "[I am willing to change my energy consumption behavior to be more environmentally friendly in the foreseeable future.] Please indicate how much you agree or disagree with the following statements."
#data[, 92] <- factor(data[, 92], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[92] <- "G03Q15_SQ023"
# LimeSurvey Field type: A
data[, 93] <- as.character(data[, 93])
attributes(data)$variable.labels[93] <- "[I am willing to sacrifice some of my comfort to be more environmentally friendly.] Please indicate how much you agree or disagree with the following statements."
#data[, 93] <- factor(data[, 93], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[93] <- "G03Q15_SQ024"
# LimeSurvey Field type: A
data[, 94] <- as.character(data[, 94])
attributes(data)$variable.labels[94] <- "[I want to learn how to use my smart home for better energy efficiency.] Please indicate how much you agree or disagree with the following statements."
#data[, 94] <- factor(data[, 94], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("Strongly  disagree", "Disagree", "Somewhat  disagree", "Neither agree  nor disagree", "Somewhat  agree", "Agree", "Strongly  agree"))
names(data)[94] <- "G03Q15_SQ025"
# LimeSurvey Field type: A
data[, 95] <- as.character(data[, 95])
attributes(data)$variable.labels[95] <- "[During the last twelve months, I have used my smart home to monitor my energy consumption.] How often have you used your smart home for the following activities on average during the last twelve months?"
#data[, 95] <- factor(data[, 95], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07","AO08"),labels=c("Daily", "Every  other day", "Once  a week", "Every  other week", "Once  a month", "Every  other month", "Very rarely", "Never"))
names(data)[95] <- "G04Q16_SQ001"
# LimeSurvey Field type: A
data[, 96] <- as.character(data[, 96])
attributes(data)$variable.labels[96] <- "[During the last twelve months, I have used my smart home to make sure that my lights or appliances are switched off when I don’t need them.] How often have you used your smart home for the following activities on average during the last twelve months?"
#data[, 96] <- factor(data[, 96], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07","AO08"),labels=c("Daily", "Every  other day", "Once  a week", "Every  other week", "Once  a month", "Every  other month", "Very rarely", "Never"))
names(data)[96] <- "G04Q16_SQ002"
# LimeSurvey Field type: A
data[, 97] <- as.character(data[, 97])
attributes(data)$variable.labels[97] <- "[During the last twelve months, I have actively tried to reduce my energy consumption with the help of my smart home.] How often have you used your smart home for the following activities on average during the last twelve months?"
#data[, 97] <- factor(data[, 97], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07","AO08"),labels=c("Daily", "Every  other day", "Once  a week", "Every  other week", "Once  a month", "Every  other month", "Very rarely", "Never"))
names(data)[97] <- "G04Q16_SQ003"
# LimeSurvey Field type: A
data[, 98] <- as.character(data[, 98])
attributes(data)$variable.labels[98] <- "[During the last twelve months, I have used my smart home to adjust the temperature when nobody is at home.] How often have you used your smart home for the following activities on average during the last twelve months?"
#data[, 98] <- factor(data[, 98], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07","AO08"),labels=c("Daily", "Every  other day", "Once  a week", "Every  other week", "Once  a month", "Every  other month", "Very rarely", "Never"))
names(data)[98] <- "G04Q16_SQ004"
# LimeSurvey Field type: A
data[, 99] <- as.character(data[, 99])
attributes(data)$variable.labels[99] <- "[Receiving more actionable and easier to understand information on how to save energy at home which is personalized based on my past usage] How likely or unlikely is it that the following smart home services would motivate you to reduce your energy consumption? If you are already using the service, please mark the box on the very right (\"Already in use\")."
#data[, 99] <- factor(data[, 99], levels=c("AO01","AO02","AO03","AO04","AO05","AO06"),labels=c("Very  unlikely", "Unlikely", "Neutral", "Likely", "Very  likely", "Already  in use"))
names(data)[99] <- "G04Q17_SQ001"
# LimeSurvey Field type: A
data[, 100] <- as.character(data[, 100])
attributes(data)$variable.labels[100] <- "[Receiving information on how much energy I consume compared to other people in my neighborhood, friends, or family] How likely or unlikely is it that the following smart home services would motivate you to reduce your energy consumption? If you are already using the service, please mark the box on the very right (\"Already in use\")."
#data[, 100] <- factor(data[, 100], levels=c("AO01","AO02","AO03","AO04","AO05","AO06"),labels=c("Very  unlikely", "Unlikely", "Neutral", "Likely", "Very  likely", "Already  in use"))
names(data)[100] <- "G04Q17_SQ002"
# LimeSurvey Field type: A
data[, 101] <- as.character(data[, 101])
attributes(data)$variable.labels[101] <- "[Setting goals on how much energy I want to use within a specified time frame] How likely or unlikely is it that the following smart home services would motivate you to reduce your energy consumption? If you are already using the service, please mark the box on the very right (\"Already in use\")."
#data[, 101] <- factor(data[, 101], levels=c("AO01","AO02","AO03","AO04","AO05","AO06"),labels=c("Very  unlikely", "Unlikely", "Neutral", "Likely", "Very  likely", "Already  in use"))
names(data)[101] <- "G04Q17_SQ003"
# LimeSurvey Field type: A
data[, 102] <- as.character(data[, 102])
attributes(data)$variable.labels[102] <- "[Receiving information about how much money I could save if I changed my energy consumption behavior] How likely or unlikely is it that the following smart home services would motivate you to reduce your energy consumption? If you are already using the service, please mark the box on the very right (\"Already in use\")."
#data[, 102] <- factor(data[, 102], levels=c("AO01","AO02","AO03","AO04","AO05","AO06"),labels=c("Very  unlikely", "Unlikely", "Neutral", "Likely", "Very  likely", "Already  in use"))
names(data)[102] <- "G04Q17_SQ004"
# LimeSurvey Field type: A
data[, 103] <- as.character(data[, 103])
attributes(data)$variable.labels[103] <- "[Receiving rewards, gifts, or tax deductions depending on how much energy I am saving] How likely or unlikely is it that the following smart home services would motivate you to reduce your energy consumption? If you are already using the service, please mark the box on the very right (\"Already in use\")."
#data[, 103] <- factor(data[, 103], levels=c("AO01","AO02","AO03","AO04","AO05","AO06"),labels=c("Very  unlikely", "Unlikely", "Neutral", "Likely", "Very  likely", "Already  in use"))
names(data)[103] <- "G04Q17_SQ005"
# LimeSurvey Field type: A
data[, 104] <- as.character(data[, 104])
attributes(data)$variable.labels[104] <- "[Receiving information on the effect my energy reduction would have on the environment] How likely or unlikely is it that the following smart home services would motivate you to reduce your energy consumption? If you are already using the service, please mark the box on the very right (\"Already in use\")."
#data[, 104] <- factor(data[, 104], levels=c("AO01","AO02","AO03","AO04","AO05","AO06"),labels=c("Very  unlikely", "Unlikely", "Neutral", "Likely", "Very  likely", "Already  in use"))
names(data)[104] <- "G04Q17_SQ006"
# LimeSurvey Field type: A
data[, 105] <- as.character(data[, 105])
attributes(data)$variable.labels[105] <- "[Having a more fun and game-like smart home user interface] How likely or unlikely is it that the following smart home services would motivate you to reduce your energy consumption? If you are already using the service, please mark the box on the very right (\"Already in use\")."
#data[, 105] <- factor(data[, 105], levels=c("AO01","AO02","AO03","AO04","AO05","AO06"),labels=c("Very  unlikely", "Unlikely", "Neutral", "Likely", "Very  likely", "Already  in use"))
names(data)[105] <- "G04Q17_SQ007"
# LimeSurvey Field type: A
data[, 106] <- as.character(data[, 106])
attributes(data)$variable.labels[106] <- "Have you noticed any cost savings on your energy bills since installing your smart home system?"
#data[, 106] <- factor(data[, 106], levels=c("AO01","AO02","AO03"),labels=c("Yes", "No", "I don\'t know./Not applicable."))
names(data)[106] <- "G04Q18"
# LimeSurvey Field type: A
data[, 107] <- as.character(data[, 107])
attributes(data)$variable.labels[107] <- "Since using my smart home, my energy consumption …"
#data[, 107] <- factor(data[, 107], levels=c("AO01","AO02","AO03","AO04"),labels=c("has increased.", "has decreased.", "stayed roughly the same.", "I don\'t know."))
names(data)[107] <- "G04Q19"
# LimeSurvey Field type: A
data[, 108] <- as.character(data[, 108])
attributes(data)$variable.labels[108] <- "How would you rate the overall usability of your smart home system and/or smart home energy management system?"
#data[, 108] <- factor(data[, 108], levels=c("AO01","AO02","AO03","AO04","AO05"),labels=c("Very poor", "Poor", "Okay", "Good", "Very good"))
names(data)[108] <- "G04Q20"
# LimeSurvey Field type: A
data[, 109] <- as.character(data[, 109])
attributes(data)$variable.labels[109] <- "How old are you?"
#data[, 109] <- factor(data[, 109], levels=c("AO01","AO02","AO03","AO04","AO05","AO06"),labels=c("18-24", "25-34", "35-44", "45-54", "55-64", "65+"))
names(data)[109] <- "G05Q21"
# LimeSurvey Field type: A
data[, 110] <- as.character(data[, 110])
attributes(data)$variable.labels[110] <- "What is your gender? "
#data[, 110] <- factor(data[, 110], levels=c("AO01","AO02","AO03"),labels=c("Female", "Male", "Non-binary/other"))
names(data)[110] <- "G05Q22"
# LimeSurvey Field type: A
data[, 111] <- as.character(data[, 111])
attributes(data)$variable.labels[111] <- "What is the highest level of education you have completed?"
#data[, 111] <- factor(data[, 111], levels=c("AO01","AO02","AO03","AO04","AO05","AO06"),labels=c("Elementary School", "Middle School", "High School", "Bachelor\'s degree", "Master\'s degree", "Doctoral degree/advanced degree"))
names(data)[111] <- "G05Q23"
# LimeSurvey Field type: A
data[, 112] <- as.character(data[, 112])
attributes(data)$variable.labels[112] <- "[Other] What is the highest level of education you have completed?"
names(data)[112] <- "G05Q23_other"
# LimeSurvey Field type: A
data[, 113] <- as.character(data[, 113])
attributes(data)$variable.labels[113] <- "What is your current occupation status?"
#data[, 113] <- factor(data[, 113], levels=c("AO01","AO02","AO03","AO04","AO05","AO06"),labels=c("Employed full-time", "Employed part-time", "Self-employed", "Unemployed", "Student", "Retired"))
names(data)[113] <- "G05Q24"
# LimeSurvey Field type: A
data[, 114] <- as.character(data[, 114])
attributes(data)$variable.labels[114] <- "[Other] What is your current occupation status?"
names(data)[114] <- "G05Q24_other"
# LimeSurvey Field type: F
#data[, 115] <- as.numeric(data[, 115])
attributes(data)$variable.labels[115] <- "In which country do you live?"
#data[, 115] <- factor(data[, 115], levels=c(065,0187,001,002,003,004,005,006,007,008,009,010,011,012,013,014,015,016,017,018,019,020,021,022,023,024,025,026,027,0121,029,030,031,032,033,034,035,036,037,038,039,040,041,042,043,044,045,046,047,048,049,050,051,052,053,054,055,056,057,058,059,060,061,062,063,064,066,067,068,069,070,071,072,073,074,075,076,077,078,079,080,081,082,083,084,085,086,087,088,089,090,091,092,093,094,095,096,097,098,099,0100,0101,0102,0103,0104,0105,0106,0107,0108,0109,0110,0111,0112,0154,0114,0115,0116,0117,0118,0119,0120,0183,0123,0124,0125,0126,0127,0128,0129,0130,0131,0132,0133,0134,0135,0136,0137,0138,0139,0140,0141,0142,0143,0144,0145,0146,0147,0148,0149,0150,0151,0152,0153,0155,0156,0157,0158,0159,0160,0161,0162,0163,0164,0165,0166,0167,0168,0169,0170,0171,0172,0173,0174,0175,0176,0177,0178,0179,0180,0181,0182,0184,0185,0186,0188,0189,0190,0191,0192,0193,0194,0195),labels=c("Germany", "United States of America", "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Nauru", "Côte d\'Ivoire", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros", "Congo (Congo-Brazzaville)", "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czechia (Czech Republic)", "Democratic Republic of the Congo", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini (fmr. \"Swaziland\")", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Holy See", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia", "Serbia", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar (formerly Burma)", "Namibia", "Uganda", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Korea", "North Macedonia", "Norway", "Oman", "Pakistan", "Palau", "Palestine State", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Ukraine", "United Arab Emirates", "United Kingdom", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"))
names(data)[115] <- "G05Q25"
# LimeSurvey Field type: A
data[, 116] <- as.character(data[, 116])
attributes(data)$variable.labels[116] <- "Which of the following best describes where you live?"
#data[, 116] <- factor(data[, 116], levels=c("AO01","AO02","AO03"),labels=c("City", "Town/semi-dense area", "Rural area"))
names(data)[116] <- "G05Q26"
# LimeSurvey Field type: A
data[, 117] <- as.character(data[, 117])
attributes(data)$variable.labels[117] <- "Are you a ..."
#data[, 117] <- factor(data[, 117], levels=c("AO01","AO02"),labels=c("home owner", "renter"))
names(data)[117] <- "G05Q27"
# LimeSurvey Field type: A
data[, 118] <- as.character(data[, 118])
attributes(data)$variable.labels[118] <- "[Other] Are you a ..."
names(data)[118] <- "G05Q27_other"
# LimeSurvey Field type: A
data[, 119] <- as.character(data[, 119])
attributes(data)$variable.labels[119] <- "How many people live in your household in total (including yourself)?"
#data[, 119] <- factor(data[, 119], levels=c("AO01","AO02","AO03","AO04","AO05"),labels=c("I live alone.", "Two people", "3-4 people", "5-6 people", "7+ people"))
names(data)[119] <- "G05Q28"
# LimeSurvey Field type: A
data[, 120] <- as.character(data[, 120])
attributes(data)$variable.labels[120] <- "How many of your household members are children (under 18)?"
#data[, 120] <- factor(data[, 120], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07"),labels=c("AO01", "1", "2", "3", "4", "5+", "Prefer not to say"))
names(data)[120] <- "G05Q29"
# LimeSurvey Field type: A
data[, 121] <- as.character(data[, 121])
attributes(data)$variable.labels[121] <- "What is the average monthly net income of your household (after taxes)?"
#data[, 121] <- factor(data[, 121], levels=c("AO01","AO02","AO03","AO04","AO05","AO06","AO07","AO08","AO09","AO10"),labels=c("less than 1,000 USD", "1,001-3,000 USD", "3,001-5,000 USD", "5,001-8,000 USD", "8,001-10,000 USD", "10,001-15,000 USD", "15,001-20,000 USD", "20,001-30,000 USD", "30,001-50,000 USD", "more than 50,000 USD"))
names(data)[121] <- "G05Q30"
# LimeSurvey Field type: A
data[, 122] <- as.character(data[, 122])
attributes(data)$variable.labels[122] <- "Do you have any suggestions for improvements of smart homes and/or energy management systems, or feedback about the survey, you would like to share?"
names(data)[122] <- "G05Q31"
