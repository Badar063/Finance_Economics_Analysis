import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sales_data():
    """Create monthly sales data for 2 years (24 months)"""
    np.random.seed(42)
    
    dates = []
    start_date = datetime(2023, 1, 1)
    for i in range(24):
        month_date = start_date + timedelta(days=i*30)
        dates.append(month_date.strftime('%Y-%m'))
    
    # Create seasonal pattern
    base_sales = 100000
    seasonal_pattern = np.array([0.8, 0.7, 0.9, 1.0, 1.1, 1.3, 
                                 1.5, 1.4, 1.2, 1.1, 0.9, 0.8,
                                 0.9, 0.8, 1.0, 1.1, 1.2, 1.4,
                                 1.6, 1.5, 1.3, 1.2, 1.0, 0.9])
    
    revenue = (base_sales * seasonal_pattern * 
               np.random.normal(1, 0.1, len(dates))).astype(int)
    
    # Calculate growth rate
    growth_rate = [0] + [((revenue[i] - revenue[i-1]) / revenue[i-1]) * 100 
                        for i in range(1, len(revenue))]
    
    df = pd.DataFrame({
        'month': dates,
        'revenue': revenue,
        'orders': (np.random.normal(500, 50, len(dates)) * 
                   seasonal_pattern).astype(int),
        'average_order_value': revenue / (np.random.normal(500, 50, len(dates)) * 
                                         seasonal_pattern),
        'growth_rate_percent': [round(rate, 2) for rate in growth_rate]
    })
    
    df['average_order_value'] = df['average_order_value'].round(2)
    return df

def create_product_performance():
    """Create product performance data across categories and regions"""
    np.random.seed(43)
    
    products = [f'Product_{chr(65+i)}' for i in range(10)]  # A-J
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Sports']
    regions = ['North', 'South', 'East', 'West']
    
    data = []
    for product in products:
        category = np.random.choice(categories)
        region = np.random.choice(regions)
        units_sold = np.random.randint(100, 5000)
        price = np.random.uniform(10, 500)
        revenue = units_sold * price
        profit_margin = np.random.uniform(0.1, 0.4)
        
        data.append({
            'product_id': product,
            'product_name': f'{product} {category.split()[0]}',
            'category': category,
            'region': region,
            'units_sold': units_sold,
            'price': round(price, 2),
            'revenue': round(revenue, 2),
            'profit_margin': round(profit_margin, 2),
            'customer_rating': round(np.random.uniform(3.0, 5.0), 1)
        })
    
    return pd.DataFrame(data)

def create_customer_behavior():
    """Create customer purchase behavior data"""
    np.random.seed(44)
    
    customer_ids = [f'CUST{1000+i}' for i in range(50)]
    segments = ['High-Value', 'Medium-Value', 'Low-Value']
    regions = ['North', 'South', 'East', 'West']
    
    data = []
    for cust_id in customer_ids:
        segment = np.random.choice(segments, p=[0.2, 0.5, 0.3])
        region = np.random.choice(regions)
        
        if segment == 'High-Value':
            total_spent = np.random.uniform(5000, 20000)
            purchase_frequency = np.random.randint(15, 50)
            avg_order_value = np.random.uniform(200, 500)
        elif segment == 'Medium-Value':
            total_spent = np.random.uniform(1000, 5000)
            purchase_frequency = np.random.randint(5, 15)
            avg_order_value = np.random.uniform(50, 200)
        else:
            total_spent = np.random.uniform(100, 1000)
            purchase_frequency = np.random.randint(1, 5)
            avg_order_value = np.random.uniform(20, 50)
        
        last_purchase_days = np.random.randint(1, 180)
        lifetime_days = np.random.randint(30, 720)
        
        data.append({
            'customer_id': cust_id,
            'customer_segment': segment,
            'region': region,
            'total_spent': round(total_spent, 2),
            'purchase_frequency': purchase_frequency,
            'avg_order_value': round(avg_order_value, 2),
            'last_purchase_days_ago': last_purchase_days,
            'customer_lifetime_days': lifetime_days,
            'returns_count': np.random.randint(0, 3)
        })
    
    return pd.DataFrame(data)

def create_marketing_revenue():
    """Create marketing spend vs revenue data"""
    np.random.seed(45)
    
    months = [f'2023-{str(i).zfill(2)}' for i in range(1, 13)] + \
             [f'2024-{str(i).zfill(2)}' for i in range(1, 13)]
    
    data = []
    base_revenue = 80000
    
    for i, month in enumerate(months):
        # Marketing spend with some trend
        marketing_spend = np.random.uniform(5000, 20000)
        
        # Revenue influenced by marketing with diminishing returns
        marketing_effect = marketing_spend * np.random.uniform(1.5, 3.0)
        
        # Add seasonality
        seasonal_factor = 1 + 0.3 * np.sin(i * np.pi / 6)
        
        revenue = (base_revenue + marketing_effect) * seasonal_factor * \
                  np.random.normal(1, 0.1)
        
        # Calculate ROI
        roi = ((revenue - base_revenue) / marketing_spend) * 100 if marketing_spend > 0 else 0
        
        data.append({
            'month': month,
            'marketing_spend': round(marketing_spend, 2),
            'revenue': round(revenue, 2),
            'roi_percent': round(roi, 2),
            'campaigns_count': np.random.randint(1, 5),
            'social_media_spend': round(marketing_spend * np.random.uniform(0.2, 0.4), 2),
            'email_spend': round(marketing_spend * np.random.uniform(0.1, 0.3), 2),
            'direct_marketing_spend': round(marketing_spend * np.random.uniform(0.3, 0.5), 2)
        })
    
    return pd.DataFrame(data)

def create_employee_performance():
    """Create employee performance metrics"""
    np.random.seed(46)
    
    employees = [f'Emp{100+i}' for i in range(20)]
    departments = ['Sales', 'Marketing', 'Customer Service', 'Operations']
    
    data = []
    for emp in employees:
        dept = np.random.choice(departments)
        
        if dept == 'Sales':
            sales_volume = np.random.uniform(100000, 500000)
            conversion_rate = np.random.uniform(0.1, 0.3)
            customer_satisfaction = np.random.uniform(4.0, 5.0)
        elif dept == 'Marketing':
            sales_volume = np.random.uniform(50000, 200000)
            conversion_rate = np.random.uniform(0.05, 0.15)
            customer_satisfaction = np.random.uniform(4.2, 4.8)
        else:
            sales_volume = np.random.uniform(20000, 100000)
            conversion_rate = np.random.uniform(0.02, 0.1)
            customer_satisfaction = np.random.uniform(4.5, 5.0)
        
        data.append({
            'employee_id': emp,
            'employee_name': f'Employee_{emp[3:]}',
            'department': dept,
            'sales_volume': round(sales_volume, 2),
            'conversion_rate': round(conversion_rate, 3),
            'customer_satisfaction': round(customer_satisfaction, 1),
            'deals_closed': np.random.randint(5, 100),
            'avg_deal_size': round(sales_volume / max(np.random.randint(10, 50), 1), 2),
            'tenure_months': np.random.randint(3, 60)
        })
    
    return pd.DataFrame(data)

def main():
    """Create all datasets and save to CSV files"""
    print("Creating datasets...")
    
    # Create directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    # Create and save each dataset
    sales_df = create_sales_data()
    sales_df.to_csv('data/sales_data.csv', index=False)
    print(f"Created sales_data.csv with {len(sales_df)} rows")
    
    product_df = create_product_performance()
    product_df.to_csv('data/product_performance.csv', index=False)
    print(f"Created product_performance.csv with {len(product_df)} rows")
    
    customer_df = create_customer_behavior()
    customer_df.to_csv('data/customer_behavior.csv', index=False)
    print(f"Created customer_behavior.csv with {len(customer_df)} rows")
    
    marketing_df = create_marketing_revenue()
    marketing_df.to_csv('data/marketing_revenue.csv', index=False)
    print(f"Created marketing_revenue.csv with {len(marketing_df)} rows")
    
    employee_df = create_employee_performance()
    employee_df.to_csv('data/employee_performance.csv', index=False)
    print(f"Created employee_performance.csv with {len(employee_df)} rows")
    
    print("All datasets created successfully!")
    
    # Create requirements.txt
    with open('requirements.txt', 'w') as f:
        f.write("""pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
statsmodels>=0.13.0
""")
    print("Created requirements.txt")
    
    # Create README.md
    with open('README.md', 'w') as f:
        f.write("""# Sales and Business Analysis Project

## Project Overview
This project analyzes sales and business performance across multiple dimensions including monthly trends, product performance, customer behavior, marketing effectiveness, and employee performance.

## Datasets
1. **sales_data.csv** - Monthly sales data for 24 months with revenue, orders, and growth rates
2. **product_performance.csv** - Product performance across categories and regions
3. **customer_behavior.csv** - Customer purchase behavior and segmentation
4. **marketing_revenue.csv** - Marketing spend vs revenue generation
5. **employee_performance.csv** - Employee performance metrics

## Analysis Objectives
1. Analyze monthly sales trends to identify seasonal patterns and growth rates
2. Compare product performance across different categories or regions
3. Examine customer purchase behavior and identify high-value segments
4. Study the relationship between marketing spend and revenue generation
5. Analyze employee performance metrics and identify top performers

## Installation
```bash
pip install -r requirements.txt
