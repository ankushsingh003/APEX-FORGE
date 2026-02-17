"""
Fix for the empty plots issue in the notebook.

The problem: Your plot_bivariate_num function has a bug on line 8.
It's using 'df' instead of 'df_copy' as the data parameter.

SOLUTION:
Replace this line in your notebook:
    sns.boxplot(data=df, x=target, y=column, palette="Blues")
    
With this line:
    sns.boxplot(data=df_copy, x=target, y=column, palette="Blues")

Alternatively, use this corrected function in your notebook:
"""

def plot_bivariate_num(df_copy, target, numerical_cols):
    num_plots = len(numerical_cols)
    num_rows = (num_plots + 1) // 2
    
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows*5))
    axes = axes.flatten()
    
    for i, column in enumerate(numerical_cols):
        # FIXED: Changed 'df' to 'df_copy'
        sns.boxplot(data=df_copy, x=target, y=column, palette="Blues", ax=axes[i])
        axes[i].set_title(f"{column} vs {target}")
    
    plt.tight_layout()
    plt.show()


print("""
TO FIX YOUR NOTEBOOK:

1. Find the plot_bivariate_num function in your notebook
2. On line 8, change 'data=df' to 'data=df_copy'
3. Re-run the cells

OR copy the corrected function above into your notebook.
""")
