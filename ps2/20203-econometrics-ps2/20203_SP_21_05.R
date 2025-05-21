# Group Assignment ESS, Econometrics 20203, part II
# Author: Stefano Graziosi

# Load the relevant libraries
library(skimr)       # descriptive analysis
library(DataExplorer)
library(psych)
library(corrplot)
library(GGally)
library(patchwork)
library(scales)

library(lmtest)      # model specification
library(sandwich)
library(forecast)
library(dynlm)
library(urca)
library(vars)
library(rugarch)

library(strucchange) # diagnostics
library(slider)
library(car)

library(dlm)         # time series
library(TSstudio)
library(feasts)
library(tseries)
library(zoo)
library(xts)
library(quantmod)

library(ARDL)        # ARDL

library(readr)       # datasets
library(fpp3)

library(ggthemes)    # fancy plots
library(viridisLite)
library(viridis)
library(gridExtra)
library(magrittr)
library(textab)
library(broom)
library(tibble)
library(knitr)

library(tidyverse)   # data manipulation (lubridate, dplyr, ggplot2, tidyr, tidyselect)
library(tinytex)

library(timechange)  # handle time changes

# Resolve conflicts for dplyr
library(conflicted)
conflicts_prefer(dplyr::filter)
conflicts_prefer(dplyr::select)

# Instructions:
# Specify, using EViews (or R/Matlab/Python), an appropriate econometric model for the year-on-year inflation rate
# (based on the consumer price index) of a selected country and sample period.
# Summarize results in a presentation (max 15 slides) and upload the PDF plus script by 11:00 on May 27, 2025.

# Data
# Each group creates its own dataset. Recommended source: IMF International Finance Statistics.
# Group 26: LATVIA

latvia <- X20203_dataset_final

# Inspect the dataset
dim(latvia)           # rows & cols
str(latvia)           # data types
skim(latvia)          # extended summary

# Define variable groups
gdp_vars      <- c("nGDP","rGDP","rGDP_pc","rGDP_USD","deflator")
cons_inv_vars <- c("cons","rcons","cons_GDP","inv","inv_GDP",
                   "finv","finv_GDP")
bp_vars       <- c("exports","exports_GDP","imports","imports_GDP",
                   "CA","CA_GDP","USDfx","REER")
gov_vars      <- c("govexp","govexp_GDP","govrev","govrev_GDP",
                   "govtax","govtax_GDP","govdef","govdef_GDP",
                   "govdebt","govdebt_GDP")
bc_vars       <- c("HPI","CPI","infl","unemp")
mon_vars      <- c("strate","ltrate","cbrate","M0","M1","M2")

# Missingness diagnostics
latvia %>% 
  summarise_all(~ sum(is.na(.))) %>% 
  pivot_longer(everything(), names_to = "variable", values_to = "n_missing") %>% 
  arrange(desc(n_missing))
DataExplorer::plot_missing(latvia)

# 1. Descriptive Analysis
# Numeric summaries and distributions
num_vars <- latvia %>% select(where(is.numeric)) %>% names()
psych::describe(latvia[num_vars])
DataExplorer::plot_histogram(latvia[num_vars])
DataExplorer::plot_density(latvia[num_vars])

# Boxplots for outlier detection
latvia %>% 
  pivot_longer(all_of(num_vars), names_to = "variable", values_to = "value") %>% 
  ggplot(aes(x = variable, y = value)) +
  geom_boxplot(outlier.colour = "firebrick") +
  theme_bw() +
  coord_flip() +
  labs(title = "Boxplots of Numeric Variables")

# Helper: faceted time-series plot
plot_time_facet <- function(df, vars, title, ncol = 2) {
  df %>%
    select(year, all_of(vars)) %>%
    pivot_longer(-year, names_to = "variable", values_to = "value") %>%
    ggplot(aes(x = year, y = value)) +
    geom_line(linewidth = 1) +
    facet_wrap(~ variable, scales = "free_y", ncol = ncol) +
    labs(title = title, x = "Year", y = NULL) +
    theme_minimal(base_size = 12)
}

# Generate and print time-series facets
p_gdp      <- plot_time_facet(latvia, gdp_vars,      "GDP Measures")
p_cons_inv <- plot_time_facet(latvia, cons_inv_vars, "Consumption & Investment")
p_bp       <- plot_time_facet(latvia, bp_vars,       "Balance of Payments")
p_gov      <- plot_time_facet(latvia, gov_vars,      "Government Intervention")
p_bc       <- plot_time_facet(latvia, bc_vars,       "Business Cycle Indicators")
p_mon      <- plot_time_facet(latvia, mon_vars,      "Monetary Measures")
print(p_gdp)
print(p_cons_inv)
print(p_bp)
print(p_gov)
print(p_bc)
print(p_mon)

# 2. Model Specification
# 2.1 Static Linear Regression (LRM)
model_lrm <- lm(infl ~ rGDP_pc + M2 + unemp + USDfx, data = latvia)
coeftest(model_lrm, vcov = vcovHC(model_lrm, type = "HC1"))
bptest(model_lrm)        # heteroskedasticity
dwtest(model_lrm)        # serial correlation

# 2.2 Univariate AR / ARIMA
infl_ts <- ts(latvia$infl, start = 2001, frequency = 1)
auto_fit_ar1 <- auto.arima(infl_ts, max.p = 4, max.q = 0, seasonal = TRUE)
summary(auto_fit_ar1)
checkresiduals(auto_fit_ar1)

auto_fit_ar2 <- auto.arima(infl_ts, max.p = 4, max.q = 0, seasonal = TRUE)
summary(auto_fit_ar2)
checkresiduals(auto_fit_ar2)

# 2.3 ARDL(p,q)
lat_ts <- ts(select(latvia, infl, rGDP_pc, M2, unemp, cbrate), start = 2001, frequency = 1)
model_ardl <- dynlm(infl ~ L(infl, 1) + diff(M2) + L(M2, 1), data = lat_ts)
summary(model_ardl)

# 2.4 Error-Correction Model (ECM)
coint_test <- ca.jo(ts.union(infl_ts, ts(latvia$M2, start = 2001)), type = "trace", K = 2)
summary(coint_test)
beta_hat <- coint_test@V[, 1]
ec_term   <- ts(cbind(infl_ts, ts(latvia$M2, 2001))[ ,1:2] %*% beta_hat, start = 2001)
ecm       <- dynlm(d(infl) ~ d(M2) + L(ec_term, 1), data = lat_ts)
summary(ecm)

# 2.5 Vector Autoregression (VAR)
var_data <- ts(latvia[, c("infl", "rGDP_pc", "M2")], start = 2001, frequency = 1)
lag_sel   <- VARselect(var_data, lag.max = 4, type = "const")
p         <- lag_sel$selection["AIC(n)"]
var_model <- vars::VAR(var_data, p = p, type = "const")
summary(var_model)
irf(var_model, impulse = "M2", response = "infl", n.ahead = 8) %>% plot()

# 2.6 ARIMA–GARCH
spec <- ugarchspec(
  mean.model     = list(armaOrder = c(1, 0)),
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  distribution.model = "std"
)
fit_garch <- ugarchfit(spec, infl_ts)
show(fit_garch)
plot(fit_garch, which = "all")

# 3. Diagnostic Checks

# 3.1 Static Linear Model
aic_lrm   <- AIC(model_lrm)
bic_lrm   <- BIC(model_lrm)
rmse_lrm  <- sqrt(mean(resid(model_lrm)^2))
dw_lrm    <- dwtest(model_lrm)$statistic
bp_p_lrm  <- bptest(model_lrm)$p.value
jb_lrm    <- jarque.bera.test(resid(model_lrm))
vif_lrm   <- vif(model_lrm)

# 3.2 ARIMA
aic_arima   <- AIC(auto_fit_ar1)
bic_arima   <- BIC(auto_fit_ar1)
arima_acc   <- accuracy(auto_fit_ar1)
rmse_arima  <- arima_acc["Training set", "RMSE"]
lb_p_arima  <- Box.test(residuals(auto_fit_ar1), lag = 5, type = "Ljung")$p.value
jb_arima    <- jarque.bera.test(resid(auto_fit_ar1))

# 3.3 ARDL
aic_ardl  <- AIC(model_ardl)
bic_ardl  <- BIC(model_ardl)
rmse_ardl <- sqrt(mean(resid(model_ardl)^2))
jb_lrm    <- jarque.bera.test(resid(model_ardl))
vif_lrm   <- vif(model_ardl)

# 3.4 ECM
aic_ecm   <- AIC(ecm)
bic_ecm   <- BIC(ecm)
rmse_ecm  <- sqrt(mean(resid(ecm)^2))
jb_lrm    <- jarque.bera.test(resid(ecm))
vif_lrm   <- vif(ecm)

# 3.5 VAR
aic_var    <- AIC(var_model)
bic_var    <- BIC(var_model)
infl_resid <- residuals(var_model)[, "infl"]
rmse_var   <- sqrt(mean(infl_resid^2))
stabs      <- max(Mod(polyroot(c(1, -lag_sel$selection["AIC(n)"]))))

# Summary table with diagnostic results
summary_results <- tibble(
  Model                 = c("OLS-LRM", "ARIMA(1,0,0)", "ARDL(1,1)", "ECM", "VAR"),
  AIC                   = c(aic_lrm, aic_arima, aic_ardl, aic_ecm, aic_var),
  BIC                   = c(bic_lrm, bic_arima, bic_ardl, bic_ecm, bic_var),
  RMSE                  = c(rmse_lrm, rmse_arima, rmse_ardl, rmse_ecm, rmse_var),
  `Durbin-Watson`       = c(dw_lrm, NA, NA, NA, NA),
  `BP p-value`          = c(bp_p_lrm, NA, NA, NA, NA),
  `JB p-value`          = c(
    jb_lrm$p.value,
    jb_arima$p.value,
    jarque.bera.test(resid(model_ardl))$p.value,
    jarque.bera.test(resid(ecm))$p.value,
    NA
  ),
  `VIF max`             = c(
    max(vif(model_lrm)),
    NA,
    max(vif(model_ardl)),
    max(vif(ecm)),
    NA
  ),
  `Ljung-Box p-value`   = c(NA, lb_p_arima, NA, NA, NA),
  `Max root modulus`    = c(NA, NA, NA, NA, stabs)
)

kable(summary_results, digits = 3, caption = "Diagnostic Summary for Selected Models")


# 4. Breakpoint Analysis
form <- infl ~ rGDP_pc + M2 + unemp + USDfx
lat_small <- latvia %>% select(year, infl, rGDP_pc, M2, unemp, USDfx)
bp_all <- breakpoints(form, data = lat_small, h = 0.25)
plot(bp_all)
opt_breaks <- which.min(BIC(bp_all))
bp_opt <- breakpoints(bp_all, breaks = opt_breaks)
obs_breaks <- bp_opt$breakpoints
year_breaks <- lat_small$year[obs_breaks]
cat("Estimated break year(s):", year_breaks, "\n")
lines(bp_opt)

# Chow tests
lat1 <- filter(lat_small, year <= year_break)
lat2 <- filter(lat_small, year > year_break)
T1   <- nrow(lat1)
T2   <- nrow(lat2)
k    <- 4

if (T2 > k) {
  mod_full <- lm(form, data = lat_small)
  RSS_full <- sum(resid(mod_full)^2)
  mod1     <- lm(form, data = lat1)
  RSS1     <- sum(resid(mod1)^2)
  mod2     <- lm(form, data = lat2)
  RSS2     <- sum(resid(mod2)^2)
  CH1_stat <- ((RSS_full - RSS1 - RSS2) / k) / ((RSS1 + RSS2) / (nrow(lat_small) - 2*k))
  CH1_p    <- 1 - pf(CH1_stat, df1 = k, df2 = nrow(lat_small) - 2*k)
  CH2_stat <- (RSS2 / (T2 - k)) / (RSS1 / (T1 - k))
  CH2_p    <- 1 - pf(CH2_stat, df1 = T2 - k, df2 = T1 - k)
}
e2       <- lat2$infl - predict(mod1, newdata = lat2)
CH3_stat <- ((RSS_full - RSS1) / RSS1) * ((T1 - k) / T2)
CH3_p    <- 1 - pf(CH3_stat, df1 = T2, df2 = T1 - k)
sigma1   <- RSS1 / (T1 - k)
CH4_stat <- sum(e2^2) / sigma1
CH4_p    <- 1 - pchisq(CH4_stat, df = T2)
tests <- tibble(
  Test      = c("CH1 (β-break)", "CH2 (σ² stability)", "CH3 (forecast β-stability)", "CH4 (forecast σ²)"),
  Statistic = c(CH1_stat, CH2_stat, CH3_stat, CH4_stat),
  `p-value` = c(CH1_p, CH2_p, CH3_p, CH4_p)
)
print(tests)

# Additional tests to justify the predicted structural break
## Residuals
mod_pre <- lm(infl ~ rGDP_pc + M2 + unemp + USDfx, data = filter(lat_small, year <= 2018))
mod_post <- lm(infl ~ rGDP_pc + M2 + unemp + USDfx, data = filter(lat_small, year > 2018))

summary(mod_pre)
summary(mod_post)

resid_pre <- resid(mod_pre)
resid_post <- resid(mod_post)

var_pre <- var(resid_pre)
var_post <- var(resid_post)

cat("Residual variance before 2018:", var_pre, "\n")
cat("Residual variance after 2018:", var_post, "\n")

lat_small$resid_full <- resid(lm(infl ~ rGDP_pc + M2 + unemp + USDfx, data = lat_small))

ggplot(lat_small, aes(x = year, y = resid_full)) +
  geom_line() +
  geom_vline(xintercept = 2018, linetype = "dashed", color = "red") +
  labs(title = "Regression residuals with structural break in 2018")

## Coefficients
window_size <- 10

rolling_coefs <- slide_dfr(
  .x = 1:(nrow(lat_small) - window_size + 1),
  .f = function(i) {
    df <- lat_small[i:(i + window_size - 1), ]
    coefs <- coef(lm(infl ~ rGDP_pc + M2 + unemp + USDfx, data = df))
    tibble(
      year = lat_small$year[i + window_size - 1],
      intercept = coefs[1],
      rGDP_pc = coefs[2],
      M2 = coefs[3],
      unemp = coefs[4],
      USDfx = coefs[5]
    )
  }
)

coefs_long <- pivot_longer(rolling_coefs, cols = -year, names_to = "Variable", values_to = "Coefficient")

ggplot(coefs_long, aes(x = year, y = Coefficient, color = Variable)) +
  geom_line() +
  geom_vline(xintercept = 2018, linetype = "dashed", color = "red") +
  labs(title = "Rolling Coefficient Estimates Over Time",
       y = "Estimated Coefficient") +
  theme_minimal()


# 5. Economic Hypotheses
# Discuss hypotheses regarding model parameters

# 6. Point Forecast
# Construct 1-step ahead forecasts for the last 10 years of the sample

# 7. Forecast Comparison
# Compare forecasts from preferred model with AR(2) model
