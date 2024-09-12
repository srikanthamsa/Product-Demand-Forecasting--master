# Optimizing Warehouse Demand Forecasting with ARIMA Models

Authors:  **NANDINI MAHESHWARI** & **SRIKANT HAMSA**.

---

**NOTE**:  For more details on the statistical understanding of the project kindly read *Introduction to time series and forecasting
Book by Peter J. Brockwell and Richard A. Davis, Production and Operations Analysis by Steven Nahmias and Supply Chain Engineering: Models and Applications by A. Ravi Ravindran, Donald P. Warsing, Jr.* 

---
## FORECASTING

---
## Project outline
- The objective of the project is to forecast the order demand using AUTOREGRESSIVE INTEGRATED MOVING AVERAGE model for 4 warehouses respectively.
- For this analysis we have used a modified version of the dataset from Kaggle (https://www.kaggle.com/felixzhao/productdemandforecasting/data).
- The basics of ACF, PACF, rolling mean average, rolling standard deviation and correlogram are explained in this documentation.
- By the end of the documentation you'll have a clear idea about **A**utoregressive **I**ntegrated **M**oving **A**verage or **ARIMA** model, data visualization, data analysis, statistical library functions in python and creation of interactive plots using plotly.  

---

## What makes the dataset interesting
- The dataset contains historical product demand for a manufacturing company with footprints globally. 
- The company provides thousands of products within dozens of product categories for 7 years. There are four central warehouses to ship products within the region it is responsible for.
- The data is available in the .csv format which allows us to perform the dataframe operations easily.

---

## 5 Steps towards Forecasting

---

## Introduction to ARIMA

---

- ARIMA is a forecasting technique. ARIMA– Auto Regressive Integrated Moving Average the key tool in Time Series Analysis.
- Models that relate the present value of a series to past values and past prediction errors - these are called ARIMA models.
- ARIMA models provide an approach to time series forecasting. 
- ARIMA is a forecasting technique that projects the future values of a series based entirely on its own inertia.
- Exponential smoothing and ARIMA models are the two most widely-used approaches to time series forecasting. 
- Exponential smoothing models are based on a description of trend and seasonality in the data, ARIMA models aim to describe the autocorrelations in the data.
- Its main application is in the area of short term forecasting requiring at least 40 historical data points. 
- It works best when your data exhibits a stable or consistent pattern over time with a minimum amount of outliers.
- ARIMA is usually superior to exponential smoothing techniques when the data is reasonably long and the correlation between past observations is stable.

--- 

## Program explanation

---

### Code summary

---

- The flow of the program is excuted in 2 ways.
- One is as per the flow chart and the alternative one is by using Auto Arima algorithm which is pre-installed package in Anaconda Python.
- The uploaded code is excuted in the Google Colab. 

---

## References

---

*The references are given in the structure of program.*
- On the information of how to clean the data [click here](http://chi2innovations.com/blog/discover-data-blog-series/how-clean-your-data-quickly-5-steps/)
- To learn about basic data pre-processing [click here](http://iasri.res.in/ebook/win_school_aa/notes/Data_Preprocessing.pdf)
- To learn the statistical concepts of time series and forecasting [click here](https://www.researchgate.net/file.PostFileLoader.html?id=55502f915f7f71d7a68b45df&assetKey=AS%3A273774321045510%401442284292820)
- To know about selecting particular data for rows and columns from pandas dataframe [click here](https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas)
- Why to go with [regression model](https://dss.princeton.edu/online_help/analysis/regression_intro.htm)
- What is [linear regression](https://www.statisticallysignificantconsulting.com/RegressionAnalysis.htm)
- To know [how to plot graph between 2 columns](https://stackoverflow.com/questions/17812978/how-to-plot-two-columns-of-a-pandas-data-frame-using-points)
- To know [how to change the pandas object to numeric](https://stackoverflow.com/questions/40095712/when-to-applypd-to-numeric-and-when-to-astypenp-float64-in-python)
- To learn about time series plots from plotly [click here](https://plot.ly/python/time-series/)
- To learn about normalizing one column in a dataframe [click here](https://stackoverflow.com/questions/28576540/how-can-i-normalize-the-data-in-a-range-of-columns-in-my-pandas-dataframe)
- Click to know [how to drop Nan objects in pandas dataframe](https://stackoverflow.com/questions/36370839/better-way-to-drop-nan-rows-in-pandas)
- To know about `groupby` and finding mean [click here](https://stackoverflow.com/questions/30482071/how-to-calculate-mean-values-grouped-on-another-column-in-pandas)
- **To know how to use offline plots in plotly [click here](https://stackoverflow.com/questions/35315726/visualize-plotly-charts-in-spyder)**
- To learn about moving average [click here](https://www.investopedia.com/ask/answers/013015/what-are-main-advantages-and-disadvantages-using-simple-moving-average-sma.asp)

---

## Why this project is good

---

- When it comes to ARIMA forecasting not all material available on internet is for Python. Many are for R. This documentation is covers the whole concept of ARIMA starting from the basics to the advanced level for Python environment.
- References are provided in such a way that there won't be any need for you to refer any other links or websites other than this documentation
- Nearly we deal with more than 1 million data in Pandas dataframe efficiently using the functions built for certain tasks.
- Last but not least we have provided a way to save the plots offline.

---

## Suggestions

---

- Since the dataset includes 4 warehouse locations, if there are location coordinates provided for warehouse we can plot them in the map and predict the demand based on the location.
- This could be extended to the suppliers and customers involved with the 4 warehouses and their product demands. If customer-A wants to buy a product or stock a product, based on the existing forecast data, location we can provide which warehouse would be better option.
- This could be further extended to Q,R model to predict the re-order point and EOQ for the inventory storage.
- If we have the customer data we can build ABC or continous review model. 
