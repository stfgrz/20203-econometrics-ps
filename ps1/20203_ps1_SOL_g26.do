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

if ("`user'" == "stefanograziosi") {
	cd "/Users/stefanograziosi/Documents/GitHub/20203-econometrics-ps/ps1"
    global filepath "/Users/stefanograziosi/Documents/GitHub/20203-econometrics-ps/ps1"
}

if ("`user'" == "sofiapisu") {
    global filepath "CAMBIA"
}

if ("`user'" == "ozguncavaser") {
    global filepath "CHANGE"
}

*=============================================================================
/* 								Tasks	 									*/
*=============================================================================

use "https://raw.githubusercontent.com/stfgrz/20203-econometrics-ps/ed2024b1967e03c401d890669b4aed2ceab793a5/ps1/assignment_data_group_26.dta", clear

*=============================================================================
/* 								Question 1 									*/
*=============================================================================

describe

summarize n w k ys

tabulate sector

histogram sector, discrete

*=============================================================================
/* 								Question 2 									*/
*=============================================================================

regress n w k
estimates store base_model

coefplot base_model, drop(_cons) title("Coefficient Plot: Base Model")

*=============================================================================
/* 								Question 3 									*/
*=============================================================================

test w
test k

test w k

testparm w k

*=============================================================================
/* 								Question 4 									*/
*=============================================================================

/* As a first solution, we can set up a fixed effects (FE) model as follows: */

regress n w k i.sector
estimates store sector_fe

testparm i.sector

anova n i.sector

/* As an alternative, we can also add interaction terms for differential slopes */

regress n w k i.sector##(c.w c.k)
estimates store sector_interact

testparm i.sector#c.w i.sector#c.k

anova n i.sector#c.w

anova n i.sector#c.k

*=============================================================================
/* 								Question 5 									*/
*=============================================================================

regress n w k i.sector

*=============================================================================
/* 								Question 6 									*/
*=============================================================================

estat imtest, white

estat hettest

*=============================================================================
/* 								Question 7 									*/
*=============================================================================

regress n w k i.sector, vce(robust)

regress n w k i.sector, robust

*=============================================================================
/* 								Final Checks 								*/
*=============================================================================

coefplot base_model sector_fe sector_interact, drop(_cons) ///
    title("Regression Coefficient Comparison") ///
    xlabel(, grid) legend(label(1 "Base Model") label(2 "Sector FE Model") label(3 "Sector Interact Model"))

esttab base_model sector_fe sector_interact using "tables.tex", replace tex ///
    se label title("Employment Regression Results") ///
    b(%9.3f) se(%9.3f) star(* 0.10 ** 0.05 *** 0.01)
	
	