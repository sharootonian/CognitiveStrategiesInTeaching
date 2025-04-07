# Load necessary packages
library(lme4)
library(lmerTest)
library(emmeans)
library(rstudioapi)
setwd(dirname(getActiveDocumentContext()$path))

exp3data_forR <- read.csv("~/data/preprocessed/exp3/exp3data_forR.csv")

# Convert variables to factors
exp3data_forR$group <- factor(exp3data_forR$group)
exp3data_forR$condition <- factor(exp3data_forR$condition)
exp3data_forR$subjID <- factor(exp3data_forR$subjID)


# Fit the linear mixed-effects model with random intercepts for subjects
model <- lmer(edge_normresp_U_obm ~ group * condition + (1 | subjID), data = exp3data_forR)

# View the summary of the model
summary(model)

# Perform ANOVA to test for significance of main effects and interaction
anova(model)

# Perform pairwise comparisons
emmeans_results <- emmeans(model, ~ group * condition)
pairwise_comparisons <- contrast(emmeans_results, method = "pairwise", adjust = "bonferroni")
print(pairwise_comparisons)


# Residual diagnostics
plot(model)
qqnorm(resid(model))
qqline(resid(model))

