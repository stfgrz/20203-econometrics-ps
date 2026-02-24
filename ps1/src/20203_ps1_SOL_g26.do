*============================================================================
/* Group number: 26 */
/* Group composition: Sofia Pisu, Stefano Graziosi, Ozgun Cavaser */
*============================================================================

*=============================================================================
/* 								Setup 										*/
*=============================================================================

set more off

/* For commands */
/*ssc install outreg2, replace*/
/*ssc install estout, replace*/
/*ssc install randomizr, replace*/
/*ssc install ritest, replace**/

/* For graphs & stuff */
 ssc install grstyle, replace
ssc install coefplot, replace
graph set window fontface "Lato"
grstyle init
grstyle set plain, horizontal

local user = c(username)

* We attempt to set the working directory dynamically. If running from the root of the repository, 
* we change directory to `ps1`. If running from within `ps1`, we stay there.
capture cd "ps1"
capture cd "..." // Fallback if necessary, but "use data/..." expects us to be in ps1.
* Note: Ensure Stata's working directory is set to the `/ps1/` folder before running.
global filepath "`c(pwd)'"

*=============================================================================
/* 								Tasks Setup 								*/
*=============================================================================

use "data/assignment_data_group_26.dta", clear

* Label variables for publication-quality output tables and graphs
label var n "Log Employment"
label var w "Log Relative Wage"
label var k "Log Physical Capital"
label var ys "Sectoral Output"
label var sector "Sector"
label define sec_lbl 1 "Sector 1" 2 "Sector 2" 3 "Sector 3" 4 "Sector 4"
label values sector sec_lbl


*=============================================================================
/* 								Question 1 									*/
*=============================================================================

* Describing data structure
describe

* Publication-quality summary statistics
* We use estpost to export summary stats nicely to LaTeX
cap ssc install estout, replace
estpost tabstat n w k ys, stats(count mean sd min max) columns(statistics) listwise
esttab using "output/summary_stats.tex", replace label noobs ///
    cells("count(fmt(%9.0fc)) mean(fmt(%9.3f)) sd(fmt(%9.3f)) min(fmt(%9.3f)) max(fmt(%9.3f))") ///
    title("Summary Statistics") ///
    collabels("N" "Mean" "Std. Dev." "Min" "Max") ///
    booktabs alignment(c)

* Tabulate sector frequencies
tabulate sector

* Generate a publication-quality histogram for sectoral distribution
histogram sector, discrete frequency ///
    xtitle("Sector") ytitle("Frequency") ///
    title("Distribution of Firms across Sectors") ///
    fcolor(navy%80) lcolor(navy) gap(20) ///
    addlabels addlabopts(mlabcolor(black)) 
graph export "output/sector_histogram.pdf", replace

*=============================================================================
/* 								Question 2 									*/
*=============================================================================

* Estimate OLS employment equation
regress n w k

* Store estimates for final table comparison
estimates store base_model

* Temporary debug coefficient plot
coefplot base_model, drop(_cons) ///
    title("Coefficient Plot: Base Model") ///
    yline(0, lpattern(dash) lcolor(red)) ///
    msymbol(S) mcolor(navy) ciopts(lcolor(navy))

*=============================================================================
/* 								Question 3 									*/
*=============================================================================

test w
test k
test w k

* We want to display the overall joint significance test on our final table 
* (in addition to the default F-stat which just tests if all slopes=0).
* Capture results of the test for W and K:
test w k
estadd scalar F_test_rk = r(F)
estadd scalar p_val_rk = r(p)

*=============================================================================
/* 								Question 4 									*/
*=============================================================================

/* As a first solution, we can set up a fixed effects (FE) model as follows: */
* We include sector indicators to allow for varying intercepts across sectors.
regress n w k i.sector
estimates store sector_fe
* Joint significance of sector FE:
testparm i.sector
estadd scalar F_test_sec_fe = r(F)

* ANOVA for model comparison
anova n i.sector

/* As an alternative, we can also add interaction terms for differential slopes */
* This estimates a model where both intercepts AND slopes are allowed to vary across sectors.
regress n w k i.sector##(c.w c.k)
estimates store sector_interact

* Test the joint significance of all interaction terms:
* This formally tests if the slopes differ significantly across sectors.
testparm i.sector#c.w i.sector#c.k
estadd scalar F_test_sec_int = r(F)

* We can also isolate the ANOVA of the interaction terms:
anova n i.sector#c.w
anova n i.sector#c.k

*=============================================================================
/* 								Question 5 									*/
*=============================================================================

* Proceeding with Sector Fixed Effects as our baseline accommodation of heterogeneity, 
* given the strong joint significance of sector indicators shown above.
regress n w k i.sector
estimates store final_model

*=============================================================================
/* 								Question 6 									*/
*=============================================================================

* Test the null hypothesis of conditional homoskedasticity
* White's Test for general heteroskedasticity:
estat imtest, white

* Breusch-Pagan / Cook-Weisberg test for linear forms of heteroskedasticity:
estat hettest

* Both tests typically suggest rejection of the null of homoskedasticity 
* in cross-sectional firm-level data, motivating the use of robust standard errors.

*=============================================================================
/* 								Question 7 									*/
*=============================================================================

* Estimate the model incorporating heteroskedasticity-robust standard errors
regress n w k i.sector, vce(robust)
estimates store robust_model
* We also store this for the final combined table

*=============================================================================
/* 								Final Checks & Outputs						*/
*=============================================================================

* Master Output Table: Combining all relevant models for comparison
esttab base_model sector_fe sector_interact robust_model using "output/main_regression_results.tex", replace label tex ///
    title("Employment Regression Models Comparison\label{tab:reg_results}") ///
    mtitles("Base Model" "Sector FE" "Sector Interacted" "Robust Model") ///
    b(%9.3f) se(%9.3f) star(* 0.10 ** 0.05 *** 0.01) ///
    scalars("F_test_rk Base F-Test (w,k)" "p_val_rk Base p-value" "F_test_sec_fe FE F-Test" "F_test_sec_int Int. F-Test") ///
    sfmt(%9.3f %9.3f %9.3f %9.3f) ///
    r2(%9.3f) ar2(%9.3f) obslast booktabs ///
    addnote("Standard errors in parentheses. * p < 0.10, ** p < 0.05, *** p < 0.01")

* Generate a publication-quality coefficient plot comparing the primary models
coefplot base_model sector_fe robust_model, drop(_cons *.sector) ///
    title("Regression Coefficient Comparison: Wage and Capital", size(medium)) ///
    xline(0, lpattern(dash) lcolor(gs10)) ///
    legend(label(1 "Base Model") label(2 "Sector FE Model") label(3 "Robust Model") rows(1)) ///
    ciopts(lwidth(thin)) msymbol(S D O) mcolor(navy maroon forest_green) ///
    ciopts(lcolor(navy maroon forest_green)) ///
    xlabel(, grid glpattern(dot)) ylabel(, labsize(small))
graph export "output/coef_comparison.pdf", replace