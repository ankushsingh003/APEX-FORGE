import pandas as pd

# Load your data
dataset = pd.read_csv(r"D:\mlops_hotel_booking_pred\archive (1)\booking.csv")
df_copy = dataset.copy()

# Print column names
print("=" * 80)
print("COLUMN NAMES IN YOUR DATASET")
print("=" * 80)
for i, col in enumerate(df_copy.columns):
    print(f"{i+1}. '{col}'")

# Check target column
print("\n" + "=" * 80)
print("TARGET COLUMN CHECK")
print("=" * 80)
if 'booking_status' in df_copy.columns:
    target = 'booking_status'
    print(f"✓ Found '{target}'")
elif 'booking status' in df_copy.columns:
    target = 'booking status'
    print(f"✓ Found '{target}'")
else:
    print("✗ No booking status column found!")
    target = None

if target:
    print(f"\nUnique values in {target}:")
    print(df_copy[target].value_counts())

# Define and check numerical columns
numerical_cols = ['number of adults', 'number of children', 'number of weekend nights', 
                  'number of week nights', 'lead time', 'repeated', 'P-C', 'P-not-C', 
                  'average price', 'special requests']

print("\n" + "=" * 80)
print("NUMERICAL COLUMNS CHECK")
print("=" * 80)
for col in numerical_cols:
    if col in df_copy.columns:
        print(f"✓ '{col}' exists - Type: {df_copy[col].dtype}")
    else:
        print(f"✗ '{col}' NOT FOUND")

# Convert to numeric
print("\n" + "=" * 80)
print("CONVERTING TO NUMERIC")
print("=" * 80)
for col in numerical_cols:
    if col in df_copy.columns:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        print(f"✓ {col}: {df_copy[col].dtype}")

# Sample data
print("\n" + "=" * 80)
print("SAMPLE DATA")
print("=" * 80)
print(df_copy[numerical_cols + ([target] if target else [])].head())

print("\n" + "=" * 80)
print("THE FIX FOR YOUR NOTEBOOK")
print("=" * 80)
print("""
Replace your plot_bivariate_num function with this:

def plot_bivariate_num(df_copy, target, numerical_cols):
    num_plots = len(numerical_cols)
    num_rows = (num_plots + 1) // 2
    
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows*5))
    axes = axes.flatten()
    
    for i, column in enumerate(numerical_cols):
        sns.boxplot(data=df_copy, x=target, y=column, palette="Blues", ax=axes[i])
        axes[i].set_title(f"{column} vs {target}")
    
    plt.tight_layout()
    plt.show()

KEY POINT: The line must have ax=axes[i] at the end!
""")
