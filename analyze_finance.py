
## Step 2: Finance Analysis Script

**analyze_finance.py**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FinanceEconomicsAnalyzer:
    def __init__(self):
        """Initialize the analyzer and load datasets"""
        print("Loading financial datasets...")
        self.stock_data = pd.read_csv('data/stock_prices.csv')
        self.spending_data = pd.read_csv('data/personal_spending.csv')
        self.portfolio_data = pd.read_csv('data/portfolio_performance.csv')
        self.economic_data = pd.read_csv('data/economic_indicators.csv')
        self.crypto_data = pd.read_csv('data/cryptocurrency.csv')
        
        # Convert date columns to datetime
        self.stock_data['date'] = pd.to_datetime(self.stock_data['date'])
        self.crypto_data['date'] = pd.to_datetime(self.crypto_data['date'])
        self.portfolio_data['month'] = pd.to_datetime(self.portfolio_data['month'] + '-01')
        self.economic_data['month'] = pd.to_datetime(self.economic_data['month'] + '-01')
        
        print("Data loaded successfully!")
    
    def analyze_stock_volatility(self):
        """Analyze stock price movements and calculate volatility"""
        print("\n" + "="*60)
        print("ANALYSIS 1: STOCK PRICE MOVEMENTS AND VOLATILITY")
        print("="*60)
        
        # Calculate volatility metrics for each stock
        volatility_metrics = []
        
        for symbol in self.stock_data['symbol'].unique():
            stock_df = self.stock_data[self.stock_data['symbol'] == symbol].copy()
            stock_df = stock_df.sort_values('date')
            
            # Calculate daily returns
            stock_df['daily_return'] = stock_df['close'].pct_change() * 100
            
            # Calculate volatility measures
            daily_volatility = stock_df['daily_return'].std()
            annualized_volatility = daily_volatility * np.sqrt(252)  # Trading days in a year
            
            # Rolling volatility (20-day window)
            stock_df['rolling_vol_20d'] = stock_df['daily_return'].rolling(window=20).std()
            
            # Maximum drawdown
            stock_df['cumulative_return'] = (1 + stock_df['daily_return']/100).cumprod()
            stock_df['running_max'] = stock_df['cumulative_return'].cummax()
            stock_df['drawdown'] = (stock_df['cumulative_return'] - stock_df['running_max']) / stock_df['running_max'] * 100
            max_drawdown = stock_df['drawdown'].min()
            
            # Sharpe Ratio (assuming 3% risk-free rate)
            avg_daily_return = stock_df['daily_return'].mean()
            sharpe_ratio = (avg_daily_return - (3/252)) / daily_volatility if daily_volatility > 0 else 0
            
            volatility_metrics.append({
                'symbol': symbol,
                'sector': stock_df['sector'].iloc[0],
                'daily_volatility': round(daily_volatility, 3),
                'annualized_volatility': round(annualized_volatility, 3),
                'max_drawdown': round(max_drawdown, 3),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'avg_daily_return': round(avg_daily_return, 3)
            })
        
        volatility_df = pd.DataFrame(volatility_metrics)
        
        print("\nVolatility Metrics by Stock:")
        print(volatility_df.to_string(index=False))
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Stock Volatility Analysis', fontsize=16, fontweight='bold')
        
        # 1. Stock price movements
        top_stocks = self.stock_data['symbol'].unique()[:5]
        for symbol in top_stocks:
            stock_df = self.stock_data[self.stock_data['symbol'] == symbol]
            axes[0, 0].plot(stock_df['date'], stock_df['close'], label=symbol, alpha=0.8)
        axes[0, 0].set_title('Stock Price Movements (Top 5)')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Volatility by sector
        sector_vol = volatility_df.groupby('sector')['annualized_volatility'].mean().sort_values()
        colors = plt.cm.viridis(np.linspace(0, 1, len(sector_vol)))
        axes[0, 1].barh(sector_vol.index, sector_vol.values, color=colors)
        axes[0, 1].set_title('Average Annualized Volatility by Sector')
        axes[0, 1].set_xlabel('Annualized Volatility (%)')
        
        # 3. Sharpe Ratio comparison
        axes[1, 0].bar(volatility_df['symbol'], volatility_df['sharpe_ratio'], 
                      color=['green' if x > 0 else 'red' for x in volatility_df['sharpe_ratio']])
        axes[1, 0].set_title('Sharpe Ratio by Stock')
        axes[1, 0].set_xlabel('Stock Symbol')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Drawdown analysis
        for symbol in top_stocks:
            stock_df = self.stock_data[self.stock_data['symbol'] == symbol].copy()
            stock_df = stock_df.sort_values('date')
            stock_df['returns'] = stock_df['close'].pct_change()
            stock_df['cumulative'] = (1 + stock_df['returns']).cumprod()
            stock_df['running_max'] = stock_df['cumulative'].cummax()
            stock_df['drawdown'] = (stock_df['cumulative'] - stock_df['running_max']) / stock_df['running_max']
            axes[1, 1].plot(stock_df['date'], stock_df['drawdown'] * 100, label=symbol, alpha=0.7)
        axes[1, 1].set_title('Drawdown Analysis')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Drawdown (%)')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('stock_volatility_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return volatility_df
    
    def analyze_personal_spending(self):
        """Analyze personal spending patterns by category"""
        print("\n" + "="*60)
        print("ANALYSIS 2: PERSONAL SPENDING PATTERNS")
        print("="*60)
        
        # Aggregate spending data
        monthly_spending = self.spending_data.groupby('month')['amount_spent'].sum().reset_index()
        category_spending = self.spending_data.groupby('category')['amount_spent'].sum().sort_values(ascending=False)
        
        # Calculate budget variance
        self.spending_data['budget_variance'] = (
            (self.spending_data['amount_spent'] - self.spending_data['budget_amount']) / 
            self.spending_data['budget_amount'] * 100
        )
        
        # Spending trends by category
        pivot_data = self.spending_data.pivot_table(
            index='month', 
            columns='category', 
            values='amount_spent', 
            aggfunc='sum'
        )
        
        print(f"\nTotal Spending: ${monthly_spending['amount_spent'].sum():,.2f}")
        print(f"\nAverage Monthly Spending: ${monthly_spending['amount_spent'].mean():,.2f}")
        print(f"\nTop Spending Categories:")
        for category, amount in category_spending.head(5).items():
            percentage = (amount / category_spending.sum()) * 100
            print(f"  {category}: ${amount:,.2f} ({percentage:.1f}%)")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Personal Spending Analysis', fontsize=16, fontweight='bold')
        
        # 1. Monthly total spending
        axes[0, 0].plot(range(len(monthly_spending)), monthly_spending['amount_spent'], 
                       marker='o', linewidth=2, color='blue')
        axes[0, 0].set_title('Total Monthly Spending')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Amount ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Spending by category
        top_categories = category_spending.head(8)
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_categories)))
        axes[0, 1].pie(top_categories.values, labels=top_categories.index, 
                      autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0, 1].set_title('Spending Distribution (Top 8 Categories)')
        
        # 3. Category trends over time
        top_5_categories = category_spending.head(5).index.tolist()
        for category in top_5_categories:
            category_data = self.spending_data[self.spending_data['category'] == category]
            axes[1, 0].plot(range(len(category_data)), category_data['amount_spent'], 
                          marker='.', label=category, alpha=0.8)
        axes[1, 0].set_title('Monthly Spending by Category')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Amount ($)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Budget variance analysis
        budget_variance_by_category = self.spending_data.groupby('category')['budget_variance'].mean().sort_values()
        colors = ['red' if x < 0 else 'green' for x in budget_variance_by_category.values]
        axes[1, 1].barh(budget_variance_by_category.index, budget_variance_by_category.values, color=colors)
        axes[1, 1].set_title('Average Budget Variance by Category (%)')
        axes[1, 1].set_xlabel('Variance from Budget (%)')
        axes[1, 1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('personal_spending_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.spending_data
    
    def analyze_portfolio_performance(self):
        """Compare investment portfolio performance against benchmarks"""
        print("\n" + "="*60)
        print("ANALYSIS 3: PORTFOLIO PERFORMANCE VS BENCHMARK")
        print("="*60)
        
        # Calculate cumulative returns
        self.portfolio_data['portfolio_cumulative'] = (
            (1 + self.portfolio_data['portfolio_return_percent']/100).cumprod()
        )
        self.portfolio_data['benchmark_cumulative'] = (
            (1 + self.portfolio_data['benchmark_return_percent']/100).cumprod()
        )
        
        # Calculate performance metrics
        total_months = len(self.portfolio_data)
        
        # Annualized returns
        portfolio_total_return = (self.portfolio_data['portfolio_cumulative'].iloc[-1] - 1) * 100
        benchmark_total_return = (self.portfolio_data['benchmark_cumulative'].iloc[-1] - 1) * 100
        
        portfolio_annualized = ((1 + portfolio_total_return/100) ** (12/total_months) - 1) * 100
        benchmark_annualized = ((1 + benchmark_total_return/100) ** (12/total_months) - 1) * 100
        
        # Risk metrics
        portfolio_volatility = self.portfolio_data['portfolio_return_percent'].std() * np.sqrt(12)
        benchmark_volatility = self.portfolio_data['benchmark_return_percent'].std() * np.sqrt(12)
        
        # Sharpe Ratio (assuming 3% risk-free rate)
        portfolio_sharpe = (portfolio_annualized - 3) / portfolio_volatility if portfolio_volatility > 0 else 0
        benchmark_sharpe = (benchmark_annualized - 3) / benchmark_volatility if benchmark_volatility > 0 else 0
        
        # Information Ratio
        excess_returns = self.portfolio_data['excess_return_percent']
        tracking_error = excess_returns.std() * np.sqrt(12)
        information_ratio = excess_returns.mean() / tracking_error * np.sqrt(12) if tracking_error > 0 else 0
        
        # Maximum drawdown
        self.portfolio_data['portfolio_drawdown'] = (
            (self.portfolio_data['portfolio_cumulative'] - 
             self.portfolio_data['portfolio_cumulative'].cummax()) / 
            self.portfolio_data['portfolio_cumulative'].cummax() * 100
        )
        max_drawdown = self.portfolio_data['portfolio_drawdown'].min()
        
        print("\nPortfolio Performance Metrics:")
        print(f"Total Return (Portfolio): {portfolio_total_return:.2f}%")
        print(f"Total Return (Benchmark): {benchmark_total_return:.2f}%")
        print(f"Annualized Return (Portfolio): {portfolio_annualized:.2f}%")
        print(f"Annualized Return (Benchmark): {benchmark_annualized:.2f}%")
        print(f"Excess Return: {portfolio_total_return - benchmark_total_return:.2f}%")
        print(f"Portfolio Volatility (Annualized): {portfolio_volatility:.2f}%")
        print(f"Benchmark Volatility (Annualized): {benchmark_volatility:.2f}%")
        print(f"Sharpe Ratio (Portfolio): {portfolio_sharpe:.3f}")
        print(f"Sharpe Ratio (Benchmark): {benchmark_sharpe:.3f}")
        print(f"Information Ratio: {information_ratio:.3f}")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Portfolio Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Cumulative returns comparison
        axes[0, 0].plot(self.portfolio_data['month'], self.portfolio_data['portfolio_cumulative'], 
                       label='Portfolio', linewidth=2, color='green')
        axes[0, 0].plot(self.portfolio_data['month'], self.portfolio_data['benchmark_cumulative'], 
                       label='Benchmark', linewidth=2, color='blue', alpha=0.7)
        axes[0, 0].fill_between(self.portfolio_data['month'], 
                               self.portfolio_data['portfolio_cumulative'],
                               self.portfolio_data['benchmark_cumulative'],
                               where=self.portfolio_data['portfolio_cumulative'] > self.portfolio_data['benchmark_cumulative'],
                               color='green', alpha=0.2, label='Outperformance')
        axes[0, 0].set_title('Cumulative Returns: Portfolio vs Benchmark')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Growth of $1')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Rolling 12-month returns
        self.portfolio_data['portfolio_rolling_12m'] = (
            self.portfolio_data['portfolio_return_percent'].rolling(window=12).sum()
        )
        self.portfolio_data['benchmark_rolling_12m'] = (
            self.portfolio_data['benchmark_return_percent'].rolling(window=12).sum()
        )
        axes[0, 1].plot(self.portfolio_data['month'], self.portfolio_data['portfolio_rolling_12m'], 
                       label='Portfolio', linewidth=2)
        axes[0, 1].plot(self.portfolio_data['month'], self.portfolio_data['benchmark_rolling_12m'], 
                       label='Benchmark', linewidth=2, alpha=0.7)
        axes[0, 1].set_title('Rolling 12-Month Returns')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Return (%)')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Drawdown analysis
        axes[1, 0].fill_between(self.portfolio_data['month'], 0, 
                               self.portfolio_data['portfolio_drawdown'], 
                               color='red', alpha=0.3)
        axes[1, 0].plot(self.portfolio_data['month'], self.portfolio_data['portfolio_drawdown'], 
                       color='red', linewidth=1)
        axes[1, 0].set_title('Portfolio Drawdown')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Risk-Return scatter
        metrics_summary = pd.DataFrame({
            'Return': [portfolio_annualized, benchmark_annualized],
            'Volatility': [portfolio_volatility, benchmark_volatility],
            'Sharpe': [portfolio_sharpe, benchmark_sharpe],
            'Label': ['Portfolio', 'Benchmark']
        })
        
        scatter = axes[1, 1].scatter(metrics_summary['Volatility'], metrics_summary['Return'], 
                                    s=metrics_summary['Sharpe']*200, alpha=0.6,
                                    c=metrics_summary['Sharpe'], cmap='RdYlGn')
        for i, row in metrics_summary.iterrows():
            axes[1, 1].annotate(row['Label'], (row['Volatility'], row['Return']),
                               xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_title('Risk-Return Profile')
        axes[1, 1].set_xlabel('Volatility (%)')
        axes[1, 1].set_ylabel('Annualized Return (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=axes[1, 1], label='Sharpe Ratio')
        
        plt.tight_layout()
        plt.savefig('portfolio_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.portfolio_data
    
    def analyze_economic_correlations(self):
        """Study correlation between economic indicators and unemployment"""
        print("\n" + "="*60)
        print("ANALYSIS 4: ECONOMIC INDICATORS CORRELATION")
        print("="*60)
        
        # Calculate correlations
        correlation_matrix = self.economic_data[[
            'unemployment_rate_percent',
            'gdp_growth_percent', 
            'inflation_rate_percent',
            'interest_rate_percent',
            'consumer_confidence_index',
            'manufacturing_pmi',
            'stock_market_return_percent'
        ]].corr()
        
        print("\nCorrelation Matrix (with Unemployment Rate):")
        unemployment_correlations = correlation_matrix['unemployment_rate_percent'].sort_values()
        for indicator, corr in unemployment_correlations.items():
            if indicator != 'unemployment_rate_percent':
                print(f"  {indicator}: {corr:.3f}")
        
        # Statistical tests
        print("\nRegression Analysis: Unemployment vs GDP Growth")
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            self.economic_data['gdp_growth_percent'],
            self.economic_data['unemployment_rate_percent']
        )
        print(f"  R-squared: {r_value**2:.3f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Coefficient: {slope:.3f} (for 1% GDP growth)")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Economic Indicators Analysis', fontsize=16, fontweight='bold')
        
        # 1. Unemployment vs GDP growth scatter
        scatter = axes[0, 0].scatter(self.economic_data['gdp_growth_percent'], 
                                    self.economic_data['unemployment_rate_percent'],
                                    c=self.economic_data['inflation_rate_percent'],
                                    cmap='RdYlBu', alpha=0.7, s=50)
        z = np.polyfit(self.economic_data['gdp_growth_percent'], 
                      self.economic_data['unemployment_rate_percent'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(self.economic_data['gdp_growth_percent'], 
                       p(self.economic_data['gdp_growth_percent']), 
                       "r--", alpha=0.8)
        axes[0, 0].set_title('Unemployment vs GDP Growth')
        axes[0, 0].set_xlabel('GDP Growth (%)')
        axes[0, 0].set_ylabel('Unemployment Rate (%)')
        plt.colorbar(scatter, ax=axes[0, 0], label='Inflation Rate (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Time series of key indicators
        axes[0, 1].plot(self.economic_data['month'], self.economic_data['unemployment_rate_percent'],
                       label='Unemployment', linewidth=2)
        axes[0, 1].plot(self.economic_data['month'], self.economic_data['gdp_growth_percent'],
                       label='GDP Growth', linewidth=2, alpha=0.7)
        axes[0, 1].plot(self.economic_data['month'], self.economic_data['inflation_rate_percent'],
                       label='Inflation', linewidth=2, alpha=0.7)
        axes[0, 1].set_title('Economic Indicators Over Time')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Rate (%)')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Correlation heatmap
        im = axes[1, 0].imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 0].set_title('Correlation Matrix')
        axes[1, 0].set_xticks(range(len(correlation_matrix.columns)))
        axes[1, 0].set_yticks(range(len(correlation_matrix.columns)))
        axes[1, 0].set_xticklabels([col[:15] for col in correlation_matrix.columns], rotation=45, ha='right')
        axes[1, 0].set_yticklabels([col[:15] for col in correlation_matrix.columns])
        
        # Add correlation values
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                text = axes[1, 0].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                      ha='center', va='center', color='black', fontsize=8)
        
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Leading indicators analysis
        # Calculate lagged correlations
        max_lag = 6
        lag_correlations = []
        
        for lag in range(max_lag + 1):
            if lag == 0:
                corr = self.economic_data['manufacturing_pmi'].corr(
                    self.economic_data['unemployment_rate_percent']
                )
            else:
                corr = self.economic_data['manufacturing_pmi'].shift(lag).corr(
                    self.economic_data['unemployment_rate_percent']
                )
            lag_correlations.append(corr)
        
        axes[1, 1].bar(range(max_lag + 1), lag_correlations, color='steelblue')
        axes[1, 1].set_title('Leading Indicator Analysis: PMI vs Unemployment')
        axes[1, 1].set_xlabel('Lag (months)')
        axes[1, 1].set_ylabel('Correlation')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('economic_indicators_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlation_matrix
    
    def analyze_cryptocurrency_trends(self):
        """Examine cryptocurrency price trends and trading volumes"""
        print("\n" + "="*60)
        print("ANALYSIS 5: CRYPTOCURRENCY TRENDS AND VOLUMES")
        print("="*60)
        
        # Calculate metrics for each cryptocurrency
        crypto_metrics = []
        
        for crypto in self.crypto_data['cryptocurrency'].unique():
            crypto_df = self.crypto_data[self.crypto_data['cryptocurrency'] == crypto].copy()
            crypto_df = crypto_df.sort_values('date')
            
            # Calculate returns and volatility
            crypto_df['returns'] = crypto_df['close'].pct_change() * 100
            
            # Basic metrics
            total_return = ((crypto_df['close'].iloc[-1] - crypto_df['close'].iloc[0]) / 
                          crypto_df['close'].iloc[0]) * 100
            volatility = crypto_df['returns'].std() * np.sqrt(365)  # Annualized
            avg_daily_volume = crypto_df['volume'].mean()
            
            # Maximum drawdown
            crypto_df['cumulative'] = (1 + crypto_df['returns']/100).cumprod()
            crypto_df['running_max'] = crypto_df['cumulative'].cummax()
            crypto_df['drawdown'] = (crypto_df['cumulative'] - crypto_df['running_max']) / crypto_df['running_max'] * 100
            max_drawdown = crypto_df['drawdown'].min()
            
            # Volume-price correlation
            volume_price_corr = crypto_df['volume'].corr(crypto_df['returns'])
            
            crypto_metrics.append({
                'cryptocurrency': crypto,
                'category': crypto_df['category'].iloc[0],
                'current_price': crypto_df['close'].iloc[-1],
                'total_return_percent': round(total_return, 2),
                'annualized_volatility': round(volatility, 2),
                'avg_daily_volume': int(avg_daily_volume),
                'max_drawdown': round(max_drawdown, 2),
                'volume_price_correlation': round(volume_price_corr, 3)
            })
        
        metrics_df = pd.DataFrame(crypto_metrics)
        
        print("\nCryptocurrency Performance Metrics:")
        print(metrics_df.to_string(index=False))
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cryptocurrency Analysis', fontsize=16, fontweight='bold')
        
        # 1. Price trends for top cryptocurrencies
        top_cryptos = metrics_df.nlargest(5, 'current_price')['cryptocurrency'].tolist()
        for crypto in top_cryptos:
            crypto_df = self.crypto_data[self.crypto_data['cryptocurrency'] == crypto]
            axes[0, 0].plot(crypto_df['date'], crypto_df['close'], label=crypto, alpha=0.8)
        axes[0, 0].set_title('Price Trends (Top 5 by Current Price)')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_yscale('log')  # Log scale for better visualization
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Returns vs Volatility scatter
        scatter = axes[0, 1].scatter(metrics_df['annualized_volatility'], 
                                    metrics_df['total_return_percent'],
                                    s=metrics_df['avg_daily_volume']/10000,
                                    c=metrics_df['max_drawdown'],
                                    cmap='RdYlGn_r', alpha=0.7)
        for i, row in metrics_df.iterrows():
            axes[0, 1].annotate(row['cryptocurrency'], 
                               (row['annualized_volatility'], row['total_return_percent']),
                               fontsize=8, alpha=0.7)
        axes[0, 1].set_title('Risk-Return Profile')
        axes[0, 1].set_xlabel('Annualized Volatility (%)')
        axes[0, 1].set_ylabel('Total Return (%)')
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 1], label='Max Drawdown (%)')
        
        # 3. Volume analysis
        volume_by_crypto = metrics_df.sort_values('avg_daily_volume', ascending=False)
        colors = plt.cm.plasma(np.linspace(0, 1, len(volume_by_crypto)))
        axes[1, 0].barh(volume_by_crypto['cryptocurrency'], 
                       volume_by_crypto['avg_daily_volume'], 
                       color=colors)
        axes[1, 0].set_title('Average Daily Trading Volume')
        axes[1, 0].set_xlabel('Volume')
        axes[1, 0].set_xscale('log')  # Log scale due to wide range
        
        # 4. Price-volume relationship for Bitcoin
        btc_data = self.crypto_data[self.crypto_data['cryptocurrency'] == 'BTC'].copy()
        btc_data = btc_data.sort_values('date')
        
        # Calculate moving averages
        btc_data['price_ma_7d'] = btc_data['close'].rolling(window=7).mean()
        btc_data['volume_ma_7d'] = btc_data['volume'].rolling(window=7).mean()
        
        ax1 = axes[1, 1]
        ax2 = ax1.twinx()
        
        ax1.plot(btc_data['date'], btc_data['price_ma_7d'], 
                color='blue', label='Price (7-day MA)', linewidth=2)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax2.plot(btc_data['date'], btc_data['volume_ma_7d'], 
                color='red', alpha=0.5, label='Volume (7-day MA)')
        ax2.set_ylabel('Volume', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_yscale('log')
        
        axes[1, 1].set_title('Bitcoin: Price vs Volume Trend')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig('cryptocurrency_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return metrics_df
    
    def run_all_analysis(self):
        """Run all analyses and generate comprehensive report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE FINANCE & ECONOMICS ANALYSIS")
        print("="*60)
        
        results = {}
        
        print("\nRunning Stock Volatility Analysis...")
        results['stock_volatility'] = self.analyze_stock_volatility()
        
        print("\nRunning Personal Spending Analysis...")
        results['personal_spending'] = self.analyze_personal_spending()
        
        print("\nRunning Portfolio Performance Analysis...")
        results['portfolio_performance'] = self.analyze_portfolio_performance()
        
        print("\nRunning Economic Indicators Analysis...")
        results['economic_correlations'] = self.analyze_economic_correlations()
        
        print("\nRunning Cryptocurrency Analysis...")
        results['cryptocurrency_trends'] = self.analyze_cryptocurrency_trends()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("\nSummary:")
        print(f"- Stock Analysis: {len(results['stock_volatility'])} stocks analyzed")
        print(f"- Spending Analysis: {len(self.spending_data['category'].unique())} categories")
        print(f"- Portfolio Analysis: {len(self.portfolio_data)} months of performance data")
        print(f"- Economic Analysis: {len(self.economic_data.columns) - 2} indicators studied")
        print(f"- Crypto Analysis: {len(results['cryptocurrency_trends'])} cryptocurrencies analyzed")
        
        # Save summary report
        self.generate_summary_report(results)
        
        return results
    
    def generate_summary_report(self, results):
        """Generate a summary report of all analyses"""
        report = []
        report.append("="*60)
        report.append("FINANCE & ECONOMICS ANALYSIS SUMMARY REPORT")
        report.append("="*60)
        report.append("")
        
        # Stock volatility summary
        report.append("STOCK VOLATILITY ANALYSIS")
        report.append("-"*40)
        most_volatile = results['stock_volatility'].loc[results['stock_volatility']['annualized_volatility'].idxmax()]
        least_volatile = results['stock_volatility'].loc[results['stock_volatility']['annualized_volatility'].idxmin()]
        best_sharpe = results['stock_volatility'].loc[results['stock_volatility']['sharpe_ratio'].idxmax()]
        
        report.append(f"Most Volatile Stock: {most_volatile['symbol']} ({most_volatile['annualized_volatility']}%)")
        report.append(f"Least Volatile Stock: {least_volatile['symbol']} ({least_volatile['annualized_volatility']}%)")
        report.append(f"Best Risk-Adjusted Return: {best_sharpe['symbol']} (Sharpe: {best_sharpe['sharpe_ratio']})")
        report.append("")
        
        # Personal spending summary
        report.append("PERSONAL SPENDING ANALYSIS")
        report.append("-"*40)
        category_summary = self.spending_data.groupby('category')['amount_spent'].sum().nlargest(3)
        report.append("Top 3 Spending Categories:")
        for category, amount in category_summary.items():
            report.append(f"  {category}: ${amount:,.2f}")
        report.append("")
        
        # Portfolio performance summary
        report.append("PORTFOLIO PERFORMANCE ANALYSIS")
        report.append("-"*40)
        portfolio_return = results['portfolio_performance']['portfolio_return_percent'].sum()
        benchmark_return = results['portfolio_performance']['benchmark_return_percent'].sum()
        excess_return = portfolio_return - benchmark_return
        report.append(f"Portfolio Total Return: {portfolio_return:.2f}%")
        report.append(f"Benchmark Total Return: {benchmark_return:.2f}%")
        report.append(f"Excess Return: {excess_return:.2f}%")
        report.append("")
        
        # Economic correlations summary
        report.append("ECONOMIC INDICATORS ANALYSIS")
        report.append("-"*40)
        unemployment_corrs = results['economic_correlations']['unemployment_rate_percent']
        strongest_neg = unemployment_corrs[unemployment_correls < 0].idxmax()
        strongest_pos = unemployment_corrs[unemployment_correls > 0].idxmin()
        report.append(f"Strongest Negative Correlation with Unemployment: {strongest_neg}")
        report.append(f"Strongest Positive Correlation with Unemployment: {strongest_pos}")
        report.append("")
        
        # Cryptocurrency summary
        report.append("CRYPTOCURRENCY ANALYSIS")
        report.append("-"*40)
        best_performer = results['cryptocurrency_trends'].loc[results['cryptocurrency_trends']['total_return_percent'].idxmax()]
        worst_performer = results['cryptocurrency_trends'].loc[results['cryptocurrency_trends']['total_return_percent'].idxmin()]
        highest_volume = results['cryptocurrency_trends'].loc[results['cryptocurrency_trends']['avg_daily_volume'].idxmax()]
        
        report.append(f"Best Performer: {best_performer['cryptocurrency']} ({best_performer['total_return_percent']}%)")
        report.append(f"Worst Performer: {worst_performer['cryptocurrency']} ({worst_performer['total_return_percent']}%)")
        report.append(f"Highest Volume: {highest_volume['cryptocurrency']} ({highest_volume['avg_daily_volume']:,} daily avg)")
        
        # Save report to file
        with open('analysis_summary_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("\nSummary report saved to 'analysis_summary_report.txt'")
        print("Visualizations saved as PNG files:")
        print("- stock_volatility_analysis.png")
        print("- personal_spending_analysis.png")
        print("- portfolio_performance_analysis.png")
        print("- economic_indicators_analysis.png")
        print("- cryptocurrency_analysis.png")

def main():
    """Main function to run the analysis"""
    print("Initializing Finance & Economics Analyzer...")
    analyzer = FinanceEconomicsAnalyzer()
    
    # Run all analyses
    results = analyzer.run_all_analysis()
    
    print("\n" + "="*60)
    print("All analyses completed successfully!")
    print("Check the generated files for detailed results.")
    print("="*60)

if __name__ == "__main__":
    main()
