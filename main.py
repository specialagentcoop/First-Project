import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

df = pd.read_csv('csv_files_git/Auto Sales data.csv') # pd.read_csv
sales_mean = df['SALES'].mean() # apvalinimas: .round(2)
# print(f'{sales_mean: .2f}') # .2f

sum_sales_by_status = df['SALES'].groupby(df['STATUS']).sum()
# print(sum_sales_by_status)

count_of_status = df['STATUS'].value_counts()
# print(count_of_status)

df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], dayfirst=True)
# print(df['ORDERDATE'])

sorted_df = df.sort_values('ORDERDATE', ascending=True) #numetam df[], jo nereikia su sort_values
# print(sorted_df)

# suskaičiuoti skirtumą su .dff().dt.days tarp dienų tarp užsakymų. dt accessor stands for datetime.
sorted_df['DAYS_SINCE_PREVIOUS_ORDER'] = sorted_df['ORDERDATE'].diff().dt.days
sorted_df[['DAYS_SINCE_PREVIOUS_ORDER', 'ORDERDATE', 'ORDERNUMBER']].head(10)
# print(sorted_df)

monthly_sales = df.groupby(pd.Grouper(key='ORDERDATE', freq='M')).sum()
# plt.figure(figsize=(10, 6))
# monthly_sales['SALES'].plot()
# plt.title('Monthly sales over time')
# plt.xlabel('Month')
# plt.ylabel('Total Sales')
# plt.show()

decomposed = seasonal_decompose(monthly_sales['SALES'], model='additive')
plt.figure(figsize=(20,6))
decomposed.plot()
plt.show()

grouped_df = df.groupby('PRODUCTLINE').agg({'QUANTITYORDERED':'sum', 'SALES':'sum'})
grouped_df = grouped_df.sort_values('SALES', ascending=False)
grouped_df['SALES'].plot(kind='bar')
plt.title('Product-line performance based on total sale')
plt.xlabel('Product-line')
plt.xticks(rotation=20)
plt.ylabel('Total sales')
plt.show()

geo_segmentation = df.groupby(by='COUNTRY').agg({'SALES':'sum'}).sort_values(by='SALES', ascending=False)
geo_segmentation.plot(kind='bar')
plt.title('Sales by Country')
plt.xlabel('Country')
plt.ylabel('Total sales')
plt.xticks(rotation=35, fontsize=7)
plt.show()
