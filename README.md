# Objective 

The goal of this project is to analyze HR attrition data to identify the key factors contributing to employee turnover in an organization undergoing rapid growth. By visualizing patterns in employee engagement, job satisfaction, and other related metrics, this analysis aims to highlight the conditions that worsen attrition. Additionally, a reporting database will be developed for business analysts to help the organization create targeted programs and processes to reduce attrition. The results of this analysis will be presented to the leadership team to inform future strategies for improving employee retention.

# Data 

This project utilized a total of 15 CSV files. The primary dataset, titled "HR Training Data", contained 35 columns and 1,001 rows, capturing various employee-related metrics. The key column names included:

- Age
 - BusinessTravel
- DailyRate
- Department
- DistanceFromHome
- JobLevel
- HourlyRate
- MonthlyRate 

- MonthlyIncome
- JobRole
- Education
- EmployeeCount
- EmployeeNumber
- EnvironmentSatisfaction
- Gender
- JobInvolvement
- training
- JobSatisfaction
- PerformanceRating
- MaritalStatus
- NumCompaniesWorked
- Over18
- OverTime
- PercentSalaryHike
- RelationshipSatisfaction
- StandardHours

- StockOptionLevel
- TotalWorkingYears
- WorkLifeBalance
- YearsAtCompany
- YearsInCurrentRole
- YearsSinceLastPromotion
- YearsWithCurrManager
- Attrition

In addition to the main file, there were 14 smaller datasets, titled "HRData(1)" through "HRData(14)", each containing the same 35 columns as "HR Training Data", but varying in row count. The breakdown of these files is as follows:

- HRData(1) - HRData(8): 11 rows, 35 columns each
- HRData(9) & HRData(10): 1,001 rows, 35 columns each
- HRData(11) & HRData(12): 11 rows, 35 columns each
- HRData(13): 1,471 rows, 35 columns
- HRData(14): 2,471 rows, 35 columns

This data was used to analyze HR attrition patterns, helping the organization identify and address factors that contribute to employee turnover.

# Data Preparation (Python)
https://github.com/Agell9/HR-Analysis/blob/main/PYTHON-HRDATA(preprocessing)-PROJECT-ONE.ipynb

The data preparation phase for this project involved several critical steps to ensure consistency, accuracy, and readiness for further analysis. A total of 15 CSV files, including the main HR Training Data and 14 additional HR datasets, were cleaned, standardized, and merged into a single dataset using python.  Initially, the dataset included a 'Department' column with three distinct departments: Sales, Human Resources, and Research & Development. However, I found significant inconsistencies between job roles and their corresponding departments. For example, employees in the 'Sales Executive' job role were incorrectly assigned to the Human Resources department, and similar mismatches were observed across other roles. This made it clear that the 'Department' column was unreliable and did not accurately reflect the organization of the workforce.

Given these inconsistencies, I decided to remove the 'Department' column entirely and instead categorize employees by their specific job roles, which provided a more accurate representation of the data. This approach ensured that roles were categorized correctly, avoiding confusion and errors during analysis.


### Loading and Initial Inspection

- The primary dataset, HR Training Data, was loaded into a DataFrame (df) using pandas.

- The additional 14 smaller datasets, named HRData (1) through HRData (14), were loaded into individual DataFrames (df1 to df14).
  ```ruby
      HrTrainingData = pd.read_csv('HR Training Data.csv')
      df = HrTrainingData

      HRDATA1 = pd.read_csv('HRData (1).csv')
      df1 = HRDATA1
      #Repeated for df2 to df14
  ```

### Employee Number Standardization:

- One of the datasets, HRData (13), contained EmployeeNumber values in a format inconsistent with the others. The employee numbers in this file were single digits, while the rest of the files used a more standardized format (e.g., 03-1933576).
- To resolve this, the EmployeeNumber column in df13 was converted to a string, and leading zeros were added to ensure a 9-digit format.
- The numbers were formatted with a dash (XX-XXXXXXX) for consistency with the rest of the data
  ```ruby
      df13['EmployeeNumber'] = df13['EmployeeNumber'].astype(str)
      df13['EmployeeNumber'] = df13['EmployeeNumber'].apply(lambda x: 
      f'{int(x):09d}' if x.isdigit() else x)
      df13['EmployeeNumber'] = df13['EmployeeNumber'].apply(lambda x: 
      f'{x[:2]}-{x[2:]}' if len(x) == 9 else x)
  ```

### Ensuring Unique Employee Numbers Across All Datasets:

- To avoid duplicate employee numbers, each smaller dataset (df1 to df14) was compared to the main HR Training Data (df). This was done by performing an inner merge on the EmployeeNumber column and checking whether any employee numbers appeared in both datasets.

- If no matches were found, a message was printed indicating no duplicate employee numbers between the datasets. This ensured that the employee numbers in the smaller datasets were unique and did not overlap with the main dataset.
- 

```ruby
   HrData_1_through_14 = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, 
      df12, df14]
      for i, HrData_1_through_14 in enumerate(HrData_1_through_14, start=1):
      result = df.merge(HrData_1_through_14, how='inner', 
      on='EmployeeNumber')
            if result.empty:
                   print(f"No matching EmployeeNumbers between df and df{i}.")
           else:
                print(f"Matching EmployeeNumbers found between df and df{i}. Number of matches: {len(result)}")
```
### Standardizing The Gender Values 

- The values for gender were inconsistent within the files, many values were illustrated as either "Male/Female" and "M/F" this was resolved by renaming the values such as, F for Female and M for Male. 

```ruby
  df15['Gender'] = df15['Gender'].replace({
       'Female': 'F',
       'Male': 'M',
       'Maleale':'M',
       'Femaleemale': 'F' })
        print(df15['Gender'].value_counts())
```
### Merging the Datasets:

- After ensuring the employee numbers were standardized and unique across the datasets, the smaller datasets (df1 to df14) were concatenated with the main HR Training Data.
- This resulted in one large DataFrame (df15) that combined all the relevant data from the individual files.

```ruby
   MERGED_HR_TRAINING_DATA1 = pd.concat([df, df1, df2, df3, df4, df5,
                                      df6, df7, df8, df9, df10, df11,
                                      df12, df13, df14])
      df15 = MERGED_HR_TRAINING_DATA1
```
### Grouping Employees by Job role:

- Due to the inaccuracies in the 'Department' column, I transitioned to categorizing employees by their job roles rather than departments. This decision was made to address the clear mismatches, such as 'Sales Executives' being listed under Human Resources. Categorizing by job roles ensured that employees were grouped correctly according to their actual roles, eliminating confusion and improving the accuracy of future analysis.

- I created nine separate CSV files, each representing a specific job role:

```ruby
    job_roles = [
          "Manager", 
          "Sales Executive", 
          "Healthcare Representative", 
          "Manufacturing Director", 
          "Sales Representative", 
          "Human Resources", 
           "Research Director", 
           "Laboratory Technician", 
           "Research Scientist"
                 ]

       for role in job_roles:
             filtered_df = df15[df15['JobRole'] == role]
            filtered_df.to_csv(f'C:\\Users\\Alber\\Downloads\\{role}_Employees.csv', 
            index=False)

        df_hr = df15[df15['JobRole'] == 'Sales Executive']
        df_hr.to_csv(r'C:\Users\Alber\Downloads\Sales Executive_Employees.csv', 
          index=False)

        df_hr = df15[df15['JobRole'] == 'Healthcare Representative']
        df_hr.to_csv(r'C:\Users\Alber\Downloads\Healthcare 
        Representative_Employees.csv', index=False)
```

### Missing Training Data

-  During the data preparation, it was discovered that the TrainingTimesLastYear column contained several missing values and the training column, while mostly complete, also had missing values.
- the missing values in TrainingTimesLastYear corresponded to non-missing values in the training column and both columns aligned with the EmployeeNumber, making it clear that these columns needed to be merged to provide a complete training record.
- this was resolved by filling the missing data in the training column with the data contained in the TrainingTimesLastyear 

```ruby
   df15['training'].fillna(df15['TrainingTimesLastYear'], inplace=True) 
       print(df15[['EmployeeNumber', 'training', 'TrainingTimesLastYear']].head()) 
       print(df15['training'])
```

### Adding Binary Values To The Attrition Column 
- Attrition was converted into binary values (1 for Yes, 0 for No), this was done to streamline the future analysis an have numerical data for performing linear regressions. 

```ruby
  df15['Attrition_Binary'] = df15['Attrition'].apply(lambda x: 1 if x == 'Yes' else 
       0)
       df15.rename(columns={'Attrition': 'Attrition-Label'}, inplace=True)
       print(df15[['Attrition-Label', 'Attrition_Binary']].head())   
```
### Saving the Final Merged Dataset:


- After extracting each job role into its own CSV, I merged all the job roles into a single workbook in excel named Merged_Hr_Training_Data_workbook where each sheet represents a unique job role. This restructured dataset allows for better analysis based on job roles, eliminating the inconsistencies tied to the original departments.

  # Additional Data Preparation In Power Bi

In Power BI, I performed additional data preparation steps to ensure the data was more interpretable for visualization purposes. Specifically, I duplicated several columns to create labeled versions of the numerical scales for better readability in charts and reports

### Duplicated Columns 

- Job Involvement, 

- Job Satisfaction 
- Performance Rating
- Work-Life Balance

### Data Type Conversion
I converted the data type of the duplicated columns from numbers to text. This conversion allowed me to assign meaningful labels to each value in the scale, enhancing the clarity of the visualizations.

**Job Involvment (Scale 1 to 4)**

- 1 = Very Low
- 2 = Low
- 3 = High
- 4 = Very High

**Job Satisfaction (Scale 1 to 5)**

- 1 = Very Low
- 2 = Low
- 3 = Medium
- 4 = High
- 5 = Very High

**Performance Rating (Scale 1 to 5)**

- 1 = Very Low
- 2 = Low
- 3 = Medium
- 4 = High
- 5 = Very High

**Work-Life Balance (Scale: 1 to 4)**

- 1 = Poor
- 2 = Fair
- 3 = Good
- 4 = Excellent

# Analysis of Employee Attrition 
https://github.com/Agell9/HR-Analysis/blob/0580db68b1c641a98cac7682759572087b4a29b6/PYTHON-HRDATA(random-forest-ATTRITION)-PROJECT-TWO%20(1).ipynb

- One of the first trends identified through exploratory data analysis was the high variance in attrition rates across different job roles. Sales Representatives exhibited the highest attrition rate at 45.5%, while Research Scientists had the lowest at 27.7%. This insight suggests that certain roles are inherently more prone to turnover, possibly due to the stress, workload, or expectations associated with them. While Sales Representatives earned higher monthly incomes on average, this did not correlate with lower attrition, indicating that factors beyond compensation—such as job satisfaction and workload—play a critical role in employee retention.

```ruby
   from sklearn.model_selection import train_test_split
      from sklearn.linear_model import LinearRegression
      import matplotlib.pyplot as plt
      import pandas as pd

      df = pd.read_csv('Merged_HR_Training_Data1.csv')

      attrition_by_job_role = df.groupby('JobRole')['Attrition- 
      Label'].value_counts(normalize=True).unstack()
```
**Barplot of attrition rates across job roles**

   ```ruby
      attrition_by_job_role.plot(kind='bar', stacked=True, figsize=(10, 6), color= 
      ['blue', 'red'])
      plt.title('Attrition Across Job Roles')
      plt.xlabel('Job Role')
      plt.ylabel('Proportion of Employees')
      plt.legend(['No Attrition', 'Yes Attrition'], loc='upper right')
      plt.show()

      print(attrition_by_job_role)


       highest_attrition_data = df[df['JobRole'] == highest_attrition_role]
```
**Comparing the factors between job roles with the highest & lowest attrition**

```ruby
 lowest_attrition_data = df[df['JobRole'] == lowest_attrition_role]
      comparison_df = pd.DataFrame({
      'Feature': ['MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 
      'PerformanceRating'],
      'Highest Attrition': [highest_attrition_data['MonthlyIncome'].mean(),
                          highest_attrition_data['YearsAtCompany'].mean(),
                          highest_attrition_data['JobSatisfaction'].mean(),
                          highest_attrition_data['PerformanceRating'].mean()],
       'Lowest Attrition': [lowest_attrition_data['MonthlyIncome'].mean(),
                         lowest_attrition_data['YearsAtCompany'].mean(),
                         lowest_attrition_data['JobSatisfaction'].mean(),
                         lowest_attrition_data['PerformanceRating'].mean()] })
        print(comparison_df)
```

# Relationship Between Training and Job Satisfaction 
https://github.com/Agell9/HR-Analysis/blob/636fefe79dfd4c5cbad7a2e4c42b345bbb835db4/PYTHON-HRDATA(regression-satisfaction%26performance).ipynb

- A key area of interest was whether the amount of training employees received influenced their job satisfaction and performance. A linear regression model was built to measure the impact of training on job satisfaction, which yielded the following results: Intercept (Baseline Job Satisfaction): 2.21 Coefficient (Impact of Training): 0.162

```ruby
   X = df[['training']]
      y_job_satisfaction = df['JobSatisfaction']

      X_train, X_test, y_train, y_test = train_test_split(X, y_job_satisfaction, 
      test_size=0.2, random_state=42)

      y_pred_job_satisfaction = model_job_satisfaction.predict(X_test)

      print(f"Intercept: {model_job_satisfaction.intercept_}, Coefficient: 
      {model_job_satisfaction.coef_}")
```

# Relationship Between Training and Performance Rating
https://github.com/Agell9/HR-Analysis/blob/636fefe79dfd4c5cbad7a2e4c42b345bbb835db4/PYTHON-HRDATA(regression-satisfaction%26performance).ipynb

- Intercept (Baseline Performance Rating): 2.49
- Coefficient (Impact of Training): 0.114

```ruby
    y_performance_rating = df['PerformanceRating']

      y_pred_performance = model_performance.predict(X_test_perf)

      print(f"Intercept: {model_performance.intercept_}, Coefficient: 
      {model_performance.coef_}")
```

#  Predictive Models for Attrition

Two predictive models were developed to assess the likelihood of employees leaving the company:

- Logistic Regression: Achieved an accuracy of 64%.
- Random Forest Classifier: Achieved an accuracy of 68%.

The Random Forest model performed better, likely because of its ability to capture non-linear interactions between variables like JobRole, OverTime, and JobSatisfaction. Both models confirm that JobSatisfaction, OverTime, and JobRole are significant predictors of attrition.

   ```ruby
      from sklearn.linear_model import LogisticRegression
      from sklearn.ensemble import RandomForestClassifier
      from sklearn.model_selection import train_test_split
      from sklearn.metrics import accuracy_score

      X = df_encoded[['MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 
      'JobRole_Sales Representative', 'OverTime_Yes']]
      y = df_encoded['Attrition_Binary']

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
      random_state=42)
```

    
**Logistic Regression Model**

   ```ruby
      log_model = LogisticRegression(max_iter=1000)
      log_model.fit(X_train, y_train)
      log_pred = log_model.predict(X_test)
      log_accuracy = accuracy_score(y_test, log_pred)
      print(f"Logistic Regression Accuracy: {log_accuracy:.2f}")
```

**Random Forest Model**

   ```ruby
      rf_model = RandomForestClassifier(random_state=42)
      rf_model.fit(X_train, y_train)
      rf_pred = rf_model.predict(X_test)
      rf_accuracy = accuracy_score(y_test, rf_pred)
      print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
```
# Refined Predictive Model 

After running the initial Random Forest model and achieving a 68% prediction accuracy. The Model was further refined using **Hyperparameter tuning and GridsearchCV**  to achieve better performance.  A more focused set of features were created while eliminating less important variables. This resulted in a new Random Forest model with a 79% prediction accuracy, offering more reliable predictions of employee attrition.


```ruby
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
     import pandas as pd
```

```ruby
df = pd.read_csv('Merged_HR_Training_Data1.csv')
X = df[['MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 'PerformanceRating', 'OverTime', 
        'DistanceFromHome', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'YearsSinceLastPromotion', 'Age']]
y = df['Attrition-Label']


```

```ruby
X = pd.get_dummies(X, columns=['OverTime'], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

```



```ruby
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy with Additional Features (YearsSinceLastPromotion & Age): {rf_accuracy:.2f}")
```



```ruby
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```




```ruby
param_grid = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [10, 20, 30],        
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 4],    
    'bootstrap': [True, False]        
}
```




```ruby
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

```





```ruby
best_rf = grid_search.best_estimator_
best_rf_pred = best_rf.predict(X_test)
best_rf_accuracy = accuracy_score(y_test, best_rf_pred)
print(f"Random Forest Accuracy after Hyperparameter Tuning: {best_rf_accuracy:.2f}")
```





```ruby
```

# Final Feature Set 
 **Performance Rating** 
- Employees with lower performance ratings were more likely to 
leave.

**Monthly Income**
- While higher incomes helped retain employees, the impact was more 
nuanced when combined with other factors.

**Age** 
- Younger employees, especially those aged 18-26, were more likely to leave the 
company.
**Years Since Last Promotion**
- Employees who had not been promoted in 3+ years saw a 
significant increase in attrition.

**Years at Company**
- Longer tenure was generally associated with lower attrition, though 
not as impactful as promotions.

**Distance From Home** 
- Longer commutes contributed to higher attrition.
# Key Findings & Conclusion 

The primary goal of this analysis was to identify factors driving employee attrition and predict which employees are most likely to leave. Through exploratory data analysis (EDA) and predictive modeling (Logistic Regression and Random Forest), we uncovered several key insights.

# Key Insights 

### Time Since Last Promotion  
- Employees who had not been promoted in 3+ years saw a significant increase in attrition. After 
14 years, over 50% of employees who hadn’t been promoted left the company.

### Job Satisfaction and Performance vs. Training
- For each additional unit of training, job satisfaction increases by 0.162 units.
- For each additional unit of training, performance rating increases by 0.114 units.

### Attrition by Job Role


- Sales Representatives had the highest attrition rate at 45.5%, despite having one of the 
highest average monthly incomes ($7,986). This indicates that income alone is 
insufficient to prevent turnover.

-  Research Scientists had the lowest attrition rate at 27.7%, even with a lower average 
monthly income ($6,858). Their higher Job Satisfaction (2.72 vs. 2.61 for Sales 
Representatives) and Performance Rating (2.95 vs. 2.70) suggest that these factors are 
more important than income in retaining employees.

### Job Satisfaction and Overtime:

- Lower Job Satisfaction was a strong predictor of attrition. Employees with lower 
satisfaction scores, particularly in Sales and Human Resources, were more likely to leave.

- Overtime was another key factor. Employees who worked overtime were more likely to 
leave









# Conclusion

The primary goal of this analysis was to identify the key factors driving employee attrition and to 
predict which employees are at the highest risk of leaving. Although the features from the initial 
Random Forest model were only 68% accurate, using a combination of these features and the 
refined model’s improved feature set can significantly aid the HR department in developing 
effective strategies to reduce attrition. By focusing on key predictors like Performance Rating, 
Years Since Last Promotion, Monthly Income, and Age, the HR team can better understand 
which employees are most at risk of leaving the company.

# Recommendations 
### Targeted Training Programs:
- Since training has a positive impact on both job satisfaction and performance, HR should 
implement targeted training programs for roles with high attrition, such as Sales Representatives
and Human Resources.
 
- The data clearly shows that training leads to increased satisfaction and performance, which can 
help reduce turnover.
### Career Development Opportunities:

-  Establish clear career development and promotion paths, prioritizing employees who have been in 
their roles for 3 years or more. Employees without promotions for 3+ years are at significantly 
higher risk of leaving.
- Providing personalized training plans and development opportunities will help keep employees 
engaged and satisfied with their career progression.
### Work-Life Balance and Retention:


- Work-life balance initiatives, such as flexible work schedules and addressing commute-related 
stress, could help reduce attrition, particularly in roles where Distance From Home is a factor.
### Performance Recognition Programs:

- Implementing performance-based incentives that recognize employees who show 
improvement after completing training can motivate participation in professional 
development programs.
 
- This will not only improve Job Satisfaction but also increase Performance Ratings,
contributing to overall productivity and retention

# Power Bi Dashboard
  https://app.powerbi.com/view?r=eyJrIjoiZWEyZmQ4NWMtODYyMC00YmI5LWJlNjItMzEzODI1ZjQyYjBhIiwidCI6ImE0MzM3ZTM0LTk0MjktNDQxNS05YjljLTJjNTQ3NmQzYWY1ZSIsImMiOjF9

# Reports
  
[DAT430-PROJECT-ONE(preprocessing).pdf](https://github.com/user-attachments/files/17093009/DAT430-PROJECT-ONE.preprocessing.pdf)

[UPDATED-DAT430 PROJECT-TWO(analysis).pdf](https://github.com/user-attachments/files/17158263/UPDATED-DAT430.PROJECT-TWO.analysis.pdf)


# Python Codes in Jupyter Notebook 
https://github.com/Agell9/HR-Analysis/blob/main/PYTHON-HRDATA(preprocessing)-PROJECT-ONE.ipynb

https://github.com/Agell9/HR-Analysis/blob/main/PYTHON-HRDATA(regression-satisfaction%26performance).ipynb

https://github.com/Agell9/HR-Analysis/blob/main/PYTHON-HRDATA(random-forest-ATTRITION)-PROJECT-TWO%20(1).ipynb

