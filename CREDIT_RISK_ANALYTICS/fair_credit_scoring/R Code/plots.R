setwd("C:/Users/Johannes/OneDrive/Dokumente/Humboldt-Universität/Msc WI/1_4. Sem/Master Thesis II/")
rm(list = ls());gc()
set.seed(0)
options(scipen=999)

library(ggplot2)
library(grid)

# read data
df <- read.csv("4_graphs/plotdata.csv", fileEncoding="UTF-8-BOM")
colnames(df) <- c("Algorithm", "BalAcc", "fracLoans", "profit", "PpL", "IND", "SEP", "SUF", "Processor", "Criteria")

df$Processor <- factor(df$Processor, levels=c("MaxProf", "Base", "Reweigh", "DiRem", "PrejRem", "ROC", "AdvDeb", "EOP", "Meta", "Platt"))
# plotting
group.colors <- c('#9e0142','#d53e4f','#f46d43','#fdae61','#fee08b','#e6f598','#abdda4','#66c2a5','#3288bd','#5e4fa2')
group.colors3 <- c('#1a1a1a','#4d4d4d','#9e0142','#d53e4f','#f46d43','#fdae61','#276419','#7fbc41', '#053061', '#3288bd')


# IND to SEP
x = abs(df$IND); y = abs(df$SEP)
labels = data.frame(x = max(x)-0.85*max(x), y = max(y)-0.1*max(y), label = corr_eqn(x, y))
IND2SEP <- ggplot(df, aes(x = abs(IND), y=abs(SEP))) + 
            geom_smooth(colour = "grey", fill=NA,method = 'lm',  alpha = 0.7) +
            geom_point(aes(color = Processor, shape = Algorithm), alpha = 0.7, size = 2.5) +
            geom_text(data = labels, aes(x = x, y = y, label = label), size = 5, color="grey20", parse = TRUE) +
            scale_color_manual(values=group.colors3) +
            xlab("IND") +
            ylab("SEP") + theme(legend.position="none", plot.margin = margin(0.2, 0, 2.2, 0, "cm")) 
            


# IND to SUF
x = abs(df$IND); y = abs(df$SUF)
labels = data.frame(x = max(x)-0.85*max(x), y = max(y)-0.1*max(y), label = corr_eqn(x, y))
IND2SUF <- ggplot(df, aes(x = abs(IND), y=abs(SUF))) + 
            geom_smooth(colour = "grey", fill=NA,method = 'lm',  alpha = 0.7) +
            geom_text(data = labels, aes(x = x, y = y, label = label), size = 5, color="grey20", parse = TRUE) +
            geom_point(aes(color = Processor, shape = Algorithm), alpha = 0.7, size = 2.5) +
            scale_color_manual(values=group.colors3) +
            xlab("IND") +
            ylab("SUF") + theme(legend.position = 'bottom')

# SEP to SUF
x = abs(df$SEP); y = abs(df$SUF)
labels = data.frame(x = max(x)-0.85*max(x), y = max(y)-0.1*max(y), label = corr_eqn(x, y))
SEP2SUF <- ggplot(df, aes(x = abs(SEP), y=abs(SUF))) + 
            geom_smooth(colour = "grey", fill=NA,method = 'lm',  alpha = 0.7) +
            geom_text(data = labels, aes(x = x, y = y, label = label), size = 5, color="grey20", parse = TRUE) +
            geom_point(aes(color = Processor, shape = Algorithm), alpha = 0.7, size = 2.5) +
            scale_color_manual(values=group.colors3) +
            xlab("SEP") +
            ylab("SUF") + theme(legend.position="none", plot.margin = margin(0.2, 0, 2.2, 0, "cm"))
lay_out(
        list(IND2SEP, 1, 1),
        list(SEP2SUF, 1, 3),
        list(IND2SUF, 1, 2)) 

#library("gridExtra")
#myPlotList <- list(IND2SEP, IND2SUF, SEP2SUF)
#do.call(grid.arrange,   c(myPlotList, list(ncol = 3)))#


# PpL to Criteria
# IND to SEP
#x = abs(df$IND[-9]); y = df$PpL[-9]
x = abs(df$IND); y = df$PpL

labels = data.frame(x = max(x)-0.3*max(x), y = max(y)-0.02*max(y), label = corr_eqn(x, y))
IND2PpL <- ggplot(df, aes(x = abs(IND), y=PpL)) + 
  #geom_smooth(data = df[-9,], aes(x = abs(IND), y=PpL), colour = "grey", fill=NA,method = 'lm',  alpha = 0.7) +
  geom_smooth(colour = "grey", fill=NA,method = 'lm',  alpha = 0.7) +
  geom_text(data = labels, aes(x = x, y = y, label = label), size = 5, color="grey20", parse = TRUE) +
  geom_point(aes(color = Processor, shape = Algorithm), alpha = 0.7, size = 2.5) +
  scale_color_manual(values=group.colors3) +
  #scale_y_continuous(limits=c(-3000, 3000), breaks=seq(-3000,3000,1000))+
  xlab("IND") +
  ylab("profit per cblackit account in TWD") + theme(legend.position="none", plot.margin = margin(0.2, 0, 2.2, 0, "cm"))

# IND to SUF
#x = abs(df$SEP[-9]); y = df$PpL[-9]
x = abs(df$SEP); y = df$PpL

labels = data.frame(x = max(x)-0.3*max(x), y = max(y)-0.02*max(y), label = corr_eqn(x, y))
SEP2PpL <- ggplot(df, aes(x = abs(SEP), y=PpL)) + 
  #geom_smooth(data = df[-9,], aes(x = abs(IND), y=PpL), colour = "grey", fill=NA,method = 'lm',  alpha = 0.7) +
  geom_smooth(colour = "grey", fill=NA,method = 'lm',  alpha = 0.7) +
  geom_text(data = labels, aes(x = x, y = y, label = label), size = 5, color="grey20", parse = TRUE) +
  geom_point(aes(color = Processor, shape = Algorithm), alpha = 0.7, size = 2.5) +
  scale_color_manual(values=group.colors3) +
  #scale_y_continuous(limits=c(-3000, 3000), breaks=seq(-3000,3000,1000))+
  xlab("SEP") +
  ylab("profit per cblackit account in TWD") + theme(legend.position = 'bottom')

# SEP to SUF
#x = abs(df$SUF[-9]); y = df$PpL[-9]
x = abs(df$SUF); y = df$PpL

labels = data.frame(x = max(x)-0.3*max(x), y = max(y)-0.02*max(y), label = corr_eqn(x, y))
SUF2PpL <- ggplot(df, aes(x = abs(SUF), y=PpL)) + 
  #geom_smooth(data = df[-9,], aes(x = abs(IND), y=PpL), colour = "grey", fill=NA,method = 'lm',  alpha = 0.7) +
  geom_smooth(colour = "grey", fill=NA,method = 'lm',  alpha = 0.7) +
  geom_text(data = labels, aes(x = x, y = y, label = label), size = 5, color="grey20", parse = TRUE) +
  geom_point(aes(color = Processor, shape = Algorithm), alpha = 0.7, size = 2.5) +
  scale_color_manual(values=group.colors3) +
  scale_y_continuous(limits=c(-500, 3000), minor_breaks = seq(500 , 3000, 500))+
  xlab("SUF") +
  ylab("profit per cblackit account in TWD") + theme(legend.position="none", plot.margin = margin(0.2, 0, 2.2, 0, "cm"))
lay_out(
  list(IND2PpL, 1, 1),
  list(SUF2PpL, 1, 3),
  list(SEP2PpL, 1, 2)) 
# BalAcc to PpL
ggplot(df, aes(x = BalAcc, y=PpL)) + 
  geom_point(aes(color = Processor, shape = Algorithm), alpha = 0.7, size = 3.5) +
  scale_color_manual(values=group.colors3) +
  xlab("balanced accuracy") +
  ylab("profit per credit account in TWD") +
  theme(text = element_text(size=14))



# rectangle plot
library(dplyr)
df_grouped <- df %>% filter(Algorithm!= "other") %>% group_by(Processor) %>% summarise(IND_min = abs(mean(IND))-sd(IND), 
                                                              IND_max = abs(mean(IND))+sd(IND),
                                                              SEP_min = abs(mean(SEP))-sd(SEP), 
                                                              SEP_max = abs(mean(SEP))+sd(SEP),
                                                              SUF_min = abs(mean(SUF))-sd(SUF), 
                                                              SUF_max = abs(mean(SUF))+sd(SUF),
                                                              BalAcc_min = abs(mean(BalAcc))-sd(BalAcc), 
                                                              BalAcc_max = abs(mean(BalAcc))+sd(BalAcc),
                                                              PpL_min = abs(mean(PpL))-sd(PpL), 
                                                              PpL_max = abs(mean(PpL))+sd(PpL))
#temp <- with(df[df$Algorithm=="other",], cbind(Processor = as.character(Processor), IND_min = abs(IND), IND_max = abs(IND), SEP_min = abs(SEP),
#                                          SEP_max = abs(SEP), SUF_min = abs(SUF), SUF_max = abs(SUF), BalAcc_min = BalAcc, BalAcc_max = BalAcc,
#                                          PpL_min = PpL, PpL_max = PpL))
#df_grouped <- rbind(df_grouped, temp)           
group.colors2 <- c('#9e0142','#f46d43','#fdae61','#e6f598','#66c2a5','#5e4fa2')
group.colors4 <- c('#1a1a1a','#9e0142','#d53e4f','#fdae61','#7fbc41', '#3288bd')

pd <- position_dodge(0.2)

IND <- ggplot() + 
  geom_rect(df_grouped, mapping=aes(xmin=IND_min, xmax=IND_max, ymin=PpL_min, ymax=PpL_max, color = Processor, fill=Processor), position = pd, size=1, alpha=0.5) +
  #scale_y_continuous(breaks=seq(0.4,0.8,0.1)) +
  scale_x_continuous(limits=c(-0.1,0.5), breaks=seq(-0.1,0.5,0.1)) +
  scale_color_manual(values=group.colors4) +
  scale_fill_manual(values=group.colors4) +
  xlab("IND") +
  ylab("profit per credit account in TWD") + theme(legend.position="none", plot.margin = margin(0.2, 0, 2.2, 0, "cm"))

SEP <- ggplot() + 
  geom_rect(df_grouped, mapping=aes(xmin=SEP_min, xmax=SEP_max, ymin=PpL_min, ymax=PpL_max, color = Processor, fill=Processor), position = pd, size=1, alpha=0.5) +
  #scale_y_continuous(breaks=seq(0.4,0.8,0.1)) +
  scale_x_continuous(limits=c(-0.1,0.5), breaks=seq(-0.1,0.5,0.1)) +
  scale_color_manual(values=group.colors4) +
  scale_fill_manual(values=group.colors4) +
  xlab("SEP") +
  ylab("profit per credit account in TWD") + theme(legend.position="bottom")

SUF <- ggplot() + 
  geom_rect(df_grouped, mapping=aes(xmin=SUF_min, xmax=SUF_max, ymin=PpL_min, ymax=PpL_max, color = Processor, fill=Processor), position = pd, size=1, alpha=0.5) +
  #scale_y_continuous(breaks=seq(0.4,0.8,0.1)) +
  scale_x_continuous(limits=c(-0.1,0.5), breaks=seq(-0.1,0.5,0.1)) +
  scale_color_manual(values=group.colors4) +
  scale_fill_manual(values=group.colors4) +
  xlab("SUF") +
  ylab("profit per credit account in TWD") + theme(legend.position="none", plot.margin = margin(0.2, 0, 2.2, 0, "cm"))

lay_out(
  list(IND, 1, 1),
  list(SUF, 1, 3),
  list(SEP, 1, 2)) 

####################################
lay_out = function(...) {    
  x <- list(...)
  n <- max(sapply(x, function(x) max(x[[2]])))
  p <- max(sapply(x, function(x) max(x[[3]])))
  grid::pushViewport(grid::viewport(layout = grid::grid.layout(n, p)))    
  
  for (i in seq_len(length(x))) {
    print(x[[i]][[1]], vp = grid::viewport(layout.pos.row = x[[i]][[2]], 
                                           layout.pos.col = x[[i]][[3]]))
  }
} 

corr_eqn <- function(x,y, digits = 2) {
  corr_coef <- round(cor(x, y), digits = digits)
  paste("italic(r) == ", corr_coef)
}

#######################################################
# Crtieria to PpL
metric <- c(as.vector(df$IND), as.vector(df$SEP), as.vector(df$SUF))
df_criteria <- rbind(df[,-c(6:8)], df[,-c(6:8)], df[,-c(6:8)])
df_criteria$MetricValue <- metric
df_criteria$Metric <- c(rep("IND", 34), rep("SEP", 34), rep("SUF", 34))
#df_criteria <- df_criteria[df_criteria$Criteria%in%c("Independence", "Separation", "Sufficiency"),]
ggplot(df_criteria, aes(x = abs(MetricValue), y=PpL)) + 
  geom_point(aes(color = Metric), alpha = 0.7, size = 2.5) +
  geom_smooth(aes(color = Metric), method = "lm", se = FALSE) +
  #scale_color_manual(values=group.colors3) +
  xlab("Fairness Metrics") +
  ylab("profit per credit account in TWD")
