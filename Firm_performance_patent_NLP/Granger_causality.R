# Libraries
library(plm) # Panel data package - for Panel Granger test and unit root test
library(tseries) # for ADF Test
library(lmtest) # for BP test


# Read the datasets
asset_causality_min_9 <- read.csv("/Users/Melike/Desktop/Causality/asset_causality_min_9.csv", stringsAsFactors = FALSE)
asset_causality_min_10 <- read.csv("/Users/Melike/Desktop/Causality/asset_causality_min_10.csv", stringsAsFactors = FALSE)
asset_causality_min_9_50 <- read.csv("/Users/Melike/Desktop/Causality/asset_causality_min_9_50.csv", stringsAsFactors = FALSE)
asset_causality_min_10_50 <- read.csv("/Users/Melike/Desktop/Causality/asset_causality_min_10_50.csv", stringsAsFactors = FALSE)

rev_causality_min_9 <- read.csv("/Users/Melike/Desktop/Causality/rev_causality_min_9.csv", stringsAsFactors = FALSE)
rev_causality_min_10 <- read.csv("/Users/Melike/Desktop/Causality/rev_causality_min_10.csv", stringsAsFactors = FALSE)
rev_causality_min_9_50 <- read.csv("/Users/Melike/Desktop/Causality/rev_causality_min_9_50.csv", stringsAsFactors = FALSE)
rev_causality_min_10_50 <- read.csv("/Users/Melike/Desktop/Causality/rev_causality_min_10_50.csv", stringsAsFactors = FALSE)


# Panel data conversion
pd_assets_min_9 <- pdata.frame(asset_causality_min_9, index = c("company", "year")) # Setting as panel data
pd_assets_min_9_50 <- pdata.frame(asset_causality_min_9_50, index = c("company", "year")) 
pd_assets_min_10 <- pdata.frame(asset_causality_min_10, index = c("company", "year")) 
pd_assets_min_10_50 <- pdata.frame(asset_causality_min_10_50, index = c("company", "year"))

pd_rev_min_9 <- pdata.frame(rev_causality_min_9, index = c("company", "year"))
pd_rev_min_9_50 <- pdata.frame(rev_causality_min_9_50, index = c("company", "year"))
pd_rev_min_10 <- pdata.frame(rev_causality_min_10, index = c("company", "year"))
pd_rev_min_10_50 <- pdata.frame(rev_causality_min_10_50, index = c("company", "year"))

# Augmented Dickey Fuller Unit Root Test 
# No unit roots if p-value is smaller than 0.05 
adf.test(pd_assets_min_9$total_assets, k=2) # Stationary
adf.test(pd_assets_min_9$num_patents, k=2) # Stationary
adf.test(pd_assets_min_9_50$total_assets, k=2) # Stationary
adf.test(pd_assets_min_9_50$num_patents, k=2) # Stationary
adf.test(pd_assets_min_10$total_assets, k=2) # Stationary
adf.test(pd_assets_min_10$num_patents, k=2) # Stationary
adf.test(pd_assets_min_10_50$total_assets, k=2) # Stationary
adf.test(pd_assets_min_10_50$num_patents, k=2) # Stationary

adf.test(pd_rev_min_9$total_rev, k=2) # Stationary
adf.test(pd_rev_min_9$num_patents, k=2) # Stationary
adf.test(pd_rev_min_9_50$total_rev, k=2) # Stationary
adf.test(pd_rev_min_9_50$num_patents, k=2) # Stationary
adf.test(pd_rev_min_10$total_rev, k=2) # Stationary
adf.test(pd_rev_min_10$num_patents, k=2) # Stationary
adf.test(pd_rev_min_10_50$total_rev, k=2) # Stationary
adf.test(pd_rev_min_10_50$num_patents, k=2) # Stationary


# Unit Root Test with plm package
# H0 is non-stationarity
# Only for balanced panel data -> only for min 10 years of patent activity
purtest(total_assets ~ 1, data = pd_assets_min_10, index = c("company", "year"), pmax = 2, test = "madwu") # Stationary
purtest(num_patents ~ 1, data = pd_assets_min_10, index = c("company", "year"), pmax = 2, test = "madwu") # Stationary
purtest(total_assets ~ 1, data = pd_assets_min_10_50, index = c("company", "year"), pmax = 2, test = "madwu") # Stationary
purtest(num_patents ~ 1, data = pd_assets_min_10_50, index = c("company", "year"), pmax = 2, test = "madwu") # Stationary

purtest(total_rev ~ 1, data = pd_rev_min_10, index = c("company", "year"), pmax = 2, test = "madwu") # Stationary
purtest(num_patents ~ 1, data = pd_rev_min_10, index = c("company", "year"), pmax = 2, test = "madwu") # Stationary
purtest(total_rev ~ 1, data = pd_rev_min_10_50, index = c("company", "year"), pmax = 2, test = "madwu") # Stationary
purtest(num_patents ~ 1, data = pd_rev_min_10_50, index = c("company", "year"), pmax = 2, test = "madwu") # Stationary


# Breusch-Pagan Test for Heteroskedasticity 
# H0 is for the Breusch-Pagan test is homoskedasticity
bptest(total_assets ~ num_patents + factor(company), data = pd_assets_min_9, studentize=F) # There is no homoskedasticity
bptest(total_assets ~ num_patents + factor(company), data = pd_assets_min_9_50, studentize=F) # There is no homoskedasticity
bptest(total_assets ~ num_patents + factor(company), data = pd_assets_min_10, studentize=F) # There is no homoskedasticity
bptest(total_assets ~ num_patents + factor(company), data = pd_assets_min_10_50, studentize=F) # There is no homoskedasticity

bptest(total_rev ~ num_patents + factor(company), data = pd_rev_min_9, studentize=F) # There is no homoskedasticity
bptest(total_rev ~ num_patents + factor(company), data = pd_rev_min_9_50, studentize=F) # There is no homoskedasticity
bptest(total_rev ~ num_patents + factor(company), data = pd_rev_min_10, studentize=F) # There is no homoskedasticity
bptest(total_rev ~ num_patents + factor(company), data = pd_rev_min_10_50, studentize=F) # There is no homoskedasticity


# Panel Granger Causality - ASSETS
pgrangertest(total_assets ~ num_patents, data = asset_causality_min_9, test = "Ztilde") # gives the standardised statistic recommended by Dumitrescu/Hurlin (2012) for fixed T samples. 
pgrangertest(total_assets ~ num_patents, data = asset_causality_min_9_50, test = "Ztilde")

pgrangertest(total_assets ~ num_patents, data = asset_causality_min_10, test = "Ztilde") 
pgrangertest(total_assets ~ num_patents, data = asset_causality_min_10_50, test = "Ztilde")
pgrangertest(total_assets ~ num_patents, data = asset_causality_min_10_50, test = "Zbar")
pgrangertest(total_assets ~ num_patents, data = asset_causality_min_10_50, test = "Wbar")


# Panel Granger Causality - REVENUES
pgrangertest(total_rev ~ num_patents, data = rev_causality_min_9, test = "Ztilde")
pgrangertest(total_rev ~ num_patents, data = rev_causality_min_9_50, test = "Ztilde")

pgrangertest(total_rev ~ num_patents, data = rev_causality_min_10, test = "Ztilde")
pgrangertest(total_rev ~ num_patents, data = rev_causality_min_10_50, test = "Ztilde")
pgrangertest(total_rev ~ num_patents, data = rev_causality_min_10_50, test = "Zbar")
pgrangertest(total_rev ~ num_patents, data = rev_causality_min_10_50, test = "Wbar")


