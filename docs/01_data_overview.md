# Data Overview

Home Credit Default Risk dataset from Kaggle. Predict loan default risk using client and credit history data.

**Competition:** https://www.kaggle.com/competitions/home-credit-default-risk/

## Table Architecture

![home_credit_architecture](/docs/home_credit_architecture.png)


## Tables

### application_train.csv / application_test.csv
Main application table with client and loan information. Train has 122 columns, test has 121 (no TARGET).

**Identity & Target:**
- `SK_ID_CURR`: Loan ID
- `TARGET`: Default indicator (1=payment difficulties, 0=all other cases) - train only

**Loan Info:**
- `NAME_CONTRACT_TYPE`: Cash loans or revolving loans
- `AMT_INCOME_TOTAL`: Client income
- `AMT_CREDIT`: Credit amount of the loan
- `AMT_ANNUITY`: Loan annuity
- `AMT_GOODS_PRICE`: Price of goods for which loan is given

**Demographics:**
- `CODE_GENDER`: Gender
- `FLAG_OWN_CAR`: Car ownership flag
- `FLAG_OWN_REALTY`: Real estate ownership flag
- `CNT_CHILDREN`: Number of children
- `CNT_FAM_MEMBERS`: Family size
- `NAME_TYPE_SUITE`: Who accompanied client during application
- `NAME_INCOME_TYPE`: Income type (Working, Commercial associate, Pensioner, etc.)
- `NAME_EDUCATION_TYPE`: Education level
- `NAME_FAMILY_STATUS`: Marital status
- `NAME_HOUSING_TYPE`: Housing situation

**Time Features (days before application, negative values):**
- `DAYS_BIRTH`: Client age in days
- `DAYS_EMPLOYED`: Employment duration
- `DAYS_REGISTRATION`: Registration change date
- `DAYS_ID_PUBLISH`: ID document change date
- `DAYS_LAST_PHONE_CHANGE`: Phone change date
- `OWN_CAR_AGE`: Age of car

**Region & Location:**
- `REGION_POPULATION_RELATIVE`: Normalized population of region
- `REGION_RATING_CLIENT`: Rating of region (1,2,3)
- `REGION_RATING_CLIENT_W_CITY`: Rating with city consideration
- `REG_REGION_NOT_LIVE_REGION`: Permanent vs contact address mismatch (region)
- `REG_REGION_NOT_WORK_REGION`: Permanent vs work address mismatch (region)
- `LIVE_REGION_NOT_WORK_REGION`: Contact vs work address mismatch (region)
- `REG_CITY_NOT_LIVE_CITY`: Permanent vs contact address mismatch (city)
- `REG_CITY_NOT_WORK_CITY`: Permanent vs work address mismatch (city)
- `LIVE_CITY_NOT_WORK_CITY`: Contact vs work address mismatch (city)

**Employment:**
- `ORGANIZATION_TYPE`: Type of organization
- `OCCUPATION_TYPE`: Occupation

**Application Process:**
- `WEEKDAY_APPR_PROCESS_START`: Day of week application submitted
- `HOUR_APPR_PROCESS_START`: Hour of day application submitted

**Contact Info:**
- `FLAG_MOBIL`: Mobile phone provided
- `FLAG_EMP_PHONE`: Work phone provided
- `FLAG_WORK_PHONE`: Work phone provided
- `FLAG_CONT_MOBILE`: Mobile reachable
- `FLAG_PHONE`: Home phone provided
- `FLAG_EMAIL`: Email provided

**External Data Sources:**
- `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`: Normalized external credit scores

**Building Characteristics (normalized, 3 aggregations: AVG, MODE, MEDI):**
- `APARTMENTS_*`: Number of apartments
- `BASEMENTAREA_*`: Basement area
- `YEARS_BEGINEXPLUATATION_*`: Building exploitation start
- `YEARS_BUILD_*`: Building age
- `COMMONAREA_*`: Common area size
- `ELEVATORS_*`: Number of elevators
- `ENTRANCES_*`: Number of entrances
- `FLOORSMAX_*`, `FLOORSMIN_*`: Floor statistics
- `LANDAREA_*`: Land area
- `LIVINGAPARTMENTS_*`, `LIVINGAREA_*`: Living space
- `NONLIVINGAPARTMENTS_*`, `NONLIVINGAREA_*`: Non-living space
- `FONDKAPREMONT_MODE`: Building repair fund
- `HOUSETYPE_MODE`: House type
- `TOTALAREA_MODE`: Total area
- `WALLSMATERIAL_MODE`: Wall material
- `EMERGENCYSTATE_MODE`: Emergency state

**Social Circle:**
- `OBS_30_CNT_SOCIAL_CIRCLE`: Observations with 30 DPD default
- `DEF_30_CNT_SOCIAL_CIRCLE`: Defaults with 30 DPD
- `OBS_60_CNT_SOCIAL_CIRCLE`: Observations with 60 DPD default
- `DEF_60_CNT_SOCIAL_CIRCLE`: Defaults with 60 DPD

**Documents:**
- `FLAG_DOCUMENT_2` through `FLAG_DOCUMENT_21`: Document submission flags

**Credit Bureau Enquiries:**
- `AMT_REQ_CREDIT_BUREAU_HOUR`: Enquiries 1 hour before application
- `AMT_REQ_CREDIT_BUREAU_DAY`: Enquiries 1 day before (excluding 1 hour)
- `AMT_REQ_CREDIT_BUREAU_WEEK`: Enquiries 1 week before (excluding 1 day)
- `AMT_REQ_CREDIT_BUREAU_MON`: Enquiries 1 month before (excluding 1 week)
- `AMT_REQ_CREDIT_BUREAU_QRT`: Enquiries 3 months before (excluding 1 month)
- `AMT_REQ_CREDIT_BUREAU_YEAR`: Enquiries 1 year before (excluding 3 months)

### bureau.csv
Credit history from other financial institutions reported to Credit Bureau. 17 columns.

**All Columns:**
- `SK_ID_CURR`: Loan ID (links to application)
- `SK_ID_BUREAU`: Bureau credit ID (unique per bureau credit)
- `CREDIT_ACTIVE`: Credit status (Closed, Active, Sold, Bad debt)
- `CREDIT_CURRENCY`: Currency (recoded)
- `DAYS_CREDIT`: Days before application when bureau credit was applied
- `CREDIT_DAY_OVERDUE`: Current days past due
- `DAYS_CREDIT_ENDDATE`: Remaining duration of credit
- `DAYS_ENDDATE_FACT`: Days since credit ended (closed credits only)
- `AMT_CREDIT_MAX_OVERDUE`: Maximum overdue amount
- `CNT_CREDIT_PROLONG`: Number of prolongations
- `AMT_CREDIT_SUM`: Current credit amount
- `AMT_CREDIT_SUM_DEBT`: Current debt amount
- `AMT_CREDIT_SUM_LIMIT`: Credit card limit
- `AMT_CREDIT_SUM_OVERDUE`: Current overdue amount
- `CREDIT_TYPE`: Type (Consumer credit, Credit card, Mortgage, Car loan, etc.)
- `DAYS_CREDIT_UPDATE`: Days before application when info was last updated
- `AMT_ANNUITY`: Annuity amount

### bureau_balance.csv
Monthly balance snapshots of bureau credits. 3 columns.

**All Columns:**
- `SK_ID_BUREAU`: Bureau credit ID (links to bureau.csv)
- `MONTHS_BALANCE`: Month relative to application (-1 = most recent)
- `STATUS`: Payment status (C=closed, X=unknown, 0=no DPD, 1=1-30 DPD, 2=31-60 DPD, 3=61-90 DPD, 4=91-120 DPD, 5=120+ DPD or written off)

### previous_application.csv
Previous loan applications at Home Credit. 37 columns.

**All Columns:**
- `SK_ID_PREV`: Previous application ID
- `SK_ID_CURR`: Current loan ID (links to application)
- `NAME_CONTRACT_TYPE`: Contract type (Cash loans, Consumer loans, Revolving loans)
- `AMT_ANNUITY`: Annuity amount
- `AMT_APPLICATION`: Requested credit amount
- `AMT_CREDIT`: Final approved credit amount
- `AMT_DOWN_PAYMENT`: Down payment amount
- `AMT_GOODS_PRICE`: Goods price
- `WEEKDAY_APPR_PROCESS_START`: Weekday of application
- `HOUR_APPR_PROCESS_START`: Hour of application
- `FLAG_LAST_APPL_PER_CONTRACT`: Last application for contract flag
- `NFLAG_LAST_APPL_IN_DAY`: Last application in day flag
- `RATE_DOWN_PAYMENT`: Down payment rate (normalized)
- `RATE_INTEREST_PRIMARY`: Interest rate (normalized)
- `RATE_INTEREST_PRIVILEGED`: Privileged interest rate (normalized)
- `NAME_CASH_LOAN_PURPOSE`: Loan purpose (Repairs, Urgent needs, etc.)
- `NAME_CONTRACT_STATUS`: Status (Approved, Cancelled, Refused, Unused offer)
- `DAYS_DECISION`: Days before current application when decision made
- `NAME_PAYMENT_TYPE`: Payment method
- `CODE_REJECT_REASON`: Rejection reason
- `NAME_TYPE_SUITE`: Who accompanied client
- `NAME_CLIENT_TYPE`: New or repeat client
- `NAME_GOODS_CATEGORY`: Goods category
- `NAME_PORTFOLIO`: Portfolio (POS, Cash, Cards, etc.)
- `NAME_PRODUCT_TYPE`: Product type (walk-in, x-sell)
- `CHANNEL_TYPE`: Acquisition channel
- `SELLERPLACE_AREA`: Seller place area
- `NAME_SELLER_INDUSTRY`: Seller industry
- `CNT_PAYMENT`: Term of previous credit
- `NAME_YIELD_GROUP`: Interest rate group (low, normal, high)
- `PRODUCT_COMBINATION`: Product combination
- `DAYS_FIRST_DRAWING`: First disbursement date
- `DAYS_FIRST_DUE`: First due date
- `DAYS_LAST_DUE_1ST_VERSION`: Last due date (first version)
- `DAYS_LAST_DUE`: Last due date
- `DAYS_TERMINATION`: Expected termination date
- `NFLAG_INSURED_ON_APPROVAL`: Insurance requested flag

### POS_CASH_balance.csv
Monthly balance snapshots of previous POS and cash loans at Home Credit. 8 columns.

**All Columns:**
- `SK_ID_PREV`: Previous credit ID (links to previous_application)
- `SK_ID_CURR`: Current loan ID
- `MONTHS_BALANCE`: Month relative to application (-1 = most recent)
- `CNT_INSTALMENT`: Term of previous credit
- `CNT_INSTALMENT_FUTURE`: Remaining installments
- `NAME_CONTRACT_STATUS`: Contract status (Active, Completed, Signed, etc.)
- `SK_DPD`: Days past due
- `SK_DPD_DEF`: Days past due with tolerance

### credit_card_balance.csv
Monthly balance snapshots of previous credit cards at Home Credit. 23 columns.

**All Columns:**
- `SK_ID_PREV`: Previous credit ID (links to previous_application)
- `SK_ID_CURR`: Current loan ID
- `MONTHS_BALANCE`: Month relative to application (-1 = most recent)
- `AMT_BALANCE`: Current balance
- `AMT_CREDIT_LIMIT_ACTUAL`: Current credit limit
- `AMT_DRAWINGS_ATM_CURRENT`: ATM withdrawal amount this month
- `AMT_DRAWINGS_CURRENT`: Total withdrawal amount this month
- `AMT_DRAWINGS_OTHER_CURRENT`: Other withdrawal amount this month
- `AMT_DRAWINGS_POS_CURRENT`: POS withdrawal amount this month
- `AMT_INST_MIN_REGULARITY`: Minimum installment for this month
- `AMT_PAYMENT_CURRENT`: Payment amount this month
- `AMT_PAYMENT_TOTAL_CURRENT`: Total payment amount this month
- `AMT_RECEIVABLE_PRINCIPAL`: Principal receivable
- `AMT_RECIVABLE`: Amount receivable
- `AMT_TOTAL_RECEIVABLE`: Total receivable
- `CNT_DRAWINGS_ATM_CURRENT`: Number of ATM withdrawals
- `CNT_DRAWINGS_CURRENT`: Number of total withdrawals
- `CNT_DRAWINGS_OTHER_CURRENT`: Number of other withdrawals
- `CNT_DRAWINGS_POS_CURRENT`: Number of POS withdrawals
- `CNT_INSTALMENT_MATURE_CUM`: Number of paid installments
- `NAME_CONTRACT_STATUS`: Contract status
- `SK_DPD`: Days past due
- `SK_DPD_DEF`: Days past due with tolerance

### installments_payments.csv
Repayment history of previous credits at Home Credit. 8 columns.

**All Columns:**
- `SK_ID_PREV`: Previous credit ID (links to previous_application)
- `SK_ID_CURR`: Current loan ID
- `NUM_INSTALMENT_VERSION`: Installment calendar version (0 for credit cards)
- `NUM_INSTALMENT_NUMBER`: Installment number
- `DAYS_INSTALMENT`: When installment was due (relative to application)
- `DAYS_ENTRY_PAYMENT`: When installment was actually paid (relative to application)
- `AMT_INSTALMENT`: Prescribed installment amount
- `AMT_PAYMENT`: Actual payment amount

## Relationships

```
application_train/test (SK_ID_CURR)
    ├── bureau (SK_ID_CURR) -> bureau_balance (SK_BUREAU_ID)
    └── previous_application (SK_ID_CURR)
            ├── POS_CASH_balance (SK_ID_PREV)
            ├── credit_card_balance (SK_ID_PREV)
            └── installments_payments (SK_ID_PREV)
```

## Notes

- Time columns are in days, relative to application date
- Negative values in DAYS_* mean "X days before application"
- Many columns are normalized (0-1 range)
- Target variable only in train set (~8% default rate)

