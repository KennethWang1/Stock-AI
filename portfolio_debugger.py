import json
from config import PORTFOLIO_FILE, DEFAULT_INITIAL_CASH, DEFAULT_INITIAL_SHARES

def check_portfolio_sanity():
    try:
        with open(PORTFOLIO_FILE, 'r') as f:
            data = json.load(f)
        
        print("Current Portfolio:")
        print(f"   Total Value: ${data.get('total_value', 0):,.2f}")
        print(f"   Cash: ${data.get('cash', 0):,.2f}")
        print(f"   Shares: {data.get('shares', 0)}")
        print(f"   Current Price: ${data.get('current_price', 0):.4f}")
        print(f"   Timestamp: {data.get('timestamp', 'N/A')}")
        
        cash = data.get('cash', 0)
        shares = data.get('shares', 0)
        total_value = data.get('total_value', 0)
        
        issues = []
        if cash > 1_000_000:
            issues.append(f"Cash too high: ${cash:,.0f}")
        if shares > 1000:
            issues.append(f"Shares too high: {shares}")
        if total_value > 1_000_000:
            issues.append(f"Total value too high: ${total_value:,.0f}")
        
        if issues:
            print("\nIssues detected:")
            for issue in issues:
                print(f"   {issue}")
            print(f"\nConsider resetting to defaults:")
            print(f"   Cash: ${DEFAULT_INITIAL_CASH}")
            print(f"   Shares: {DEFAULT_INITIAL_SHARES}")
        else:
            print("\nPortfolio values look reasonable!")
            
    except FileNotFoundError:
        print("No portfolio file found")
    except Exception as e:
        print(f"Error reading portfolio: {e}")

if __name__ == "__main__":
    check_portfolio_sanity()
