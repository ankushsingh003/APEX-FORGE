import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
dataset = pd.read_csv(r"D:\mlops_hotel_booking_pred\archive (1)\booking.csv")
df_copy = dataset.copy()

# Print column names to verify
print("=" * 80)
print("STEP 1: Column Names")
print("=" * 80)
print(df_copy.columns.tolist())

# Check if 'booking status' or 'booking_status' exists
print("\n" + "=" * 80)
print("STEP 2: Checking Target Column")
print("=" * 80)
if 'booking_status' in df_copy.columns:
    print("✓ Found 'booking_status'")
    target = 'booking_status'
elif 'booking status' in df_copy.columns:
    print("✓ Found 'booking status'")
    target = 'booking status'
else:
    print("✗ Target column not found!")
    print("Available columns:", df_copy.columns.tolist())
    target = None

# Define numerical columns
numerical_cols = ['number of adults', 'number of children', 'number of weekend nights', 
                  'number of week nights', 'lead time', 'repeated', 'P-C', 'P-not-C', 
                  'average price', 'special requests']

# Convert to numeric
print("\n" + "=" * 80)
print("STEP 3: Converting to Numeric Types")
print("=" * 80)
for col in numerical_cols:
    if col in df_copy.columns:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        print(f"✓ {col}: {df_copy[col].dtype}")
    else:
        print(f"✗ Column '{col}' not found!")

# Check for NaN values
print("\n" + "=" * 80)
print("STEP 4: Checking for NaN values")
print("=" * 80)
for col in numerical_cols:
    if col in df_copy.columns:
        nan_count = df_copy[col].isna().sum()
        print(f"{col}: {nan_count} NaN values")

# Check target variable
if target:
    print("\n" + "=" * 80)
    print("STEP 5: Target Variable Info")
    print("=" * 80)
    print(f"Unique values in {target}:")
    print(df_copy[target].value_counts())
    print(f"\nData type: {df_copy[target].dtype}")

# CORRECTED PLOTTING FUNCTION
print("\n" + "=" * 80)
print("STEP 6: Creating Plots")
print("=" * 80)

def plot_bivariate_num_FIXED(df, target_col, numerical_cols):
    """
    Fixed version of the plotting function
    """
    # Filter only columns that exist
    valid_cols = [col for col in numerical_cols if col in df.columns]
    
    num_plots = len(valid_cols)
    num_rows = (num_plots + 1) // 2
    
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows*5))
    axes = axes.flatten()
    
    for i, column in enumerate(valid_cols):
        # KEY FIX: Must include ax=axes[i] parameter!
        sns.boxplot(data=df, x=target_col, y=column, palette="Blues", ax=axes[i])
        axes[i].set_title(f"{column} vs {target_col}")
    
    # Hide extra subplots if odd number of plots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    print(f"✓ Created {len(valid_cols)} plots")

# Try to create the plots
if target:
    try:
        plot_bivariate_num_FIXED(df_copy, target, numerical_cols)
        print("\n✓✓✓ SUCCESS! Plots created successfully!")
    except Exception as e:
        print(f"\n✗✗✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
else:
    print("\n✗ Cannot create plots - target column not found")

print("\n" + "=" * 80)
print("COPY THIS FUNCTION TO YOUR NOTEBOOK:")
print("=" * 80)
print("""
def plot_bivariate_num(df_copy, target, numerical_cols):
    num_plots = len(numerical_cols)
    num_rows = (num_plots + 1) // 2
    
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows*5))
    axes = axes.flatten()
    
    for i, column in enumerate(numerical_cols):
        # IMPORTANT: Must include ax=axes[i]
        sns.boxplot(data=df_copy, x=target, y=column, palette="Blues", ax=axes[i])
        axes[i].set_title(f"{column} vs {target}")
    
    plt.tight_layout()
    plt.show()
""")
