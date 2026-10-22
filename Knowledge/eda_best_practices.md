# EDA Best Practices

## Report structure
Every EDA report should follow this order: dataset overview (rows, columns, types), data quality (missing values, duplicates, wrong types), univariate analysis (distributions), outliers, relationships (correlations, feature vs target), and finally recommendations. Every claim must be backed by a number from the analysis. EDA describes and flags; it does not clean, delete or transform data.

## Dataset overview
Always start by reporting the number of rows and columns, the count of numeric and categorical columns, and memory size if large. Check that each column's type matches its meaning: numbers stored as text, dates stored as strings, and ID columns stored as numbers are common problems. ID columns (unique for every row) should be excluded from distribution and correlation analysis.

## Missing values
Report the percentage missing for every column, sorted from highest to lowest. Under 5% missing is usually safe to impute or to drop those rows. Between 5% and 40%, investigate whether the missingness is random by comparing other columns for rows with and without missing values. Above 40%, flag the column as a candidate for removal or for a "was_missing" indicator feature. Never silently drop missing data, and note that missingness itself can carry information.

## Duplicates
Report the number of fully duplicated rows. Duplicates can be genuine repeated events or data collection errors; flag them and recommend checking with the data owner rather than deleting them automatically. Also check for duplicates on key columns such as customer ID.

## Numeric distributions
For each numeric column, report mean, median, standard deviation, minimum, maximum and skewness, and plot a histogram. If the mean and median differ substantially, the distribution is skewed. Look for impossible values (negative ages, percentages above 100), suspicious spikes at a single value (such as 0 or 999, often used as placeholders for missing data), and columns with almost no variation.

## Skewed distributions
Absolute skewness above 1 means highly skewed; between 0.5 and 1 means moderately skewed. For skewed columns, report the median and interquartile range instead of the mean and standard deviation, because extreme values distort the mean. Suggest a log or Box-Cox transformation for later modelling. If values span several orders of magnitude, plot them on a log scale.

## Outliers
Use the IQR rule (below Q1 − 1.5×IQR or above Q3 + 1.5×IQR) for skewed data. Use the z-score rule (absolute z above 3) only for roughly normal data. Distinguish errors, which are impossible values, from genuine extremes, which are rare but real. If more than 5% of a column is flagged as outliers, the distribution is probably heavy-tailed rather than contaminated. Never remove outliers during EDA; only report and flag them.

## Categorical variables
For each categorical column, report the number of unique values and the most frequent categories with their percentages. Flag inconsistent spellings or capitalisation of the same category. Categories that cover less than 1% of rows can be grouped into "Other" for modelling.

## High-cardinality categoricals
A categorical column with more than 50 unique values, or where unique values exceed 50% of the rows, is high-cardinality. It may be an ID, free text, or need special encoding. Do not plot every category; show only the top 10 to 20.

## Correlation
Use Spearman correlation by default, because it is robust to outliers and captures monotonic non-linear relationships. Use Pearson only when both variables are roughly normal and the relationship looks linear. An absolute correlation above 0.7 indicates strong correlation and possible multicollinearity; above 0.9, one of the pair may be redundant. Always state that correlation does not imply causation. For relationships between two categorical variables, use Cramér's V.

## Target variable
If a target column is given, analyse it first. For a categorical target, report class proportions; if the smallest class is under 20% of rows, flag class imbalance. For a numeric target, check its distribution and skewness. Then examine how each feature relates to the target.

## Target leakage
Flag any feature that correlates with the target above 0.95, or that seems to be recorded after the outcome happened, as possible target leakage. Leaked features make models look excellent in testing but fail in real use.

## Recommendations
End the report with specific, prioritised next steps tied to findings: which columns to impute or drop, which to transform, which outliers need checking with the data owner, and which correlated features to review. Each recommendation should name the column and the reason.
