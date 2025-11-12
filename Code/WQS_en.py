import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import openpyxl
import warnings
import matplotlib as mpl
import matplotlib.font_manager as fm
import statsmodels.api as sm

# Set Chinese font display
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei on Windows
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign rendering

# ===================== Publication-grade visualization settings =====================
# 1. Set professional fonts (Times-like for English, SimHei for Chinese)
plt.rcParams.update({
    # Global font settings
    'font.family': 'serif',  # Primary font family
    'font.serif': ['Arial'],
    'font.size': 18,
    'font.weight': 'bold',

    # 坐标轴设置
    'axes.unicode_minus': False,
    'axes.linewidth': 2.0,
    'axes.labelweight': 'bold',
    'axes.titlesize': 20,
    'axes.titleweight': 'bold',

    # 图形设置
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'figure.titlesize': 22,
    'figure.titleweight': 'bold',

    # 刻度设置
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,

    # 图例设置
    'legend.fontsize': 16,
    'legend.frameon': True,
    'legend.framealpha': 0.8,

    # Chinese font support
    'font.sans-serif': ['SimHei']  # Chinese font support on Windows
})


# 2. Create plotting function
def create_journal_quality_plot(data, file_name, plot_type, outcome_column=None, r2_value=None):

    plt.figure(figsize=(7, 5), dpi=600)

    # Thicken axes spines
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2.0)

    # Thicken lines/markers
    plt.rc('lines', linewidth=2.5)
    plt.rc('scatter', edgecolors='k')

    # Weights plot - coefficient labels removed
    if plot_type == "weights":
        weights_df = data
        # Sort by weight (descending; heaviest at top)
        weights_df = weights_df.sort_values('Weight', ascending=True)

        # Define English labels for exposure pathways
        route_labels = {
            'soil_ingestion': 'Soil Ing',
            'soil_dermal': 'Soil Derm',
            'water_ingestion': 'Water Ing',
            'water_dermal': 'Water Derm',
            'diet_rice': 'Rice',
            'diet_solanaceous': 'Solanaceous',
            'diet_root_tuber': 'Root&Tuber',
            'diet_legumes': 'Legumes',
            'diet_other': 'Other Foods'
        }

        # Get English labels for the sorted variables
        labels = [route_labels[var] for var in weights_df['Variable']]

        # Create horizontal bar plot with error bars only
        plt.barh(labels, weights_df['Weight'],
                 xerr=[weights_df['Weight'] - weights_df['CI_lower'],
                       weights_df['CI_upper'] - weights_df['Weight']],
                 capsize=8, color='steelblue', alpha=0.8, edgecolor='k')

        plt.xlabel('Weight Value', fontsize=20, fontweight='bold')

        # Set x-axis limits to ensure bars are visible
        plt.xlim(0, min(1.0, weights_df['CI_upper'].max() * 1.2))

    # WQS index scatter plot
    elif plot_type == "scatter":
        # Create scatter plot
        sns.regplot(x=data['WQS_index'], y=data[outcome_column],
                    scatter_kws={'s': 60, 'edgecolor': 'k', 'alpha': 0.7, 'color': 'steelblue'},
                    line_kws={'linewidth': 2.5, 'color': 'firebrick'})

        plt.xlabel('Weighted Quantile Sum (WQS) Index', fontsize=20, fontweight='bold')
        plt.ylabel('Urinary Cd (log-transformed)', fontsize=20, fontweight='bold')

        # Annotate R²
        if r2_value is not None:
            plt.annotate(f'R² = {r2_value:.3f}',
                         xy=(0.05, 0.95), xycoords='axes fraction',
                         fontsize=20, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="k", lw=2))

    # Save publication format
    plt.tight_layout(pad=2.0)
    plt.savefig(f"{file_name}.tiff", format='tiff', bbox_inches='tight')
    plt.savefig(f"{file_name}.pdf", format='pdf', bbox_inches='tight')
    print(f"Saved publication-quality figure: {file_name}.tiff/pdf")
    plt.close() 


def wqs_regression(df, exposure_columns, outcome_column, covariate_columns=None,
                   q=10, b=100, validation_size=0.4, positive_weights=True):
    """
    Parameters:
    df: DataFrame containing analysis data
    exposure_columns: List of exposure variables
    outcome_column: Outcome variable
    covariate_columns: List of covariates
    q: Number of quantiles
    b: Number of bootstrap iterations
    validation_size: Validation set proportion
    positive_weights: Whether to constrain weights to be positive

    Returns:
    results: Dictionary containing weights and analysis results
    """

    # Handle covariates
    X_cov = df[covariate_columns] if covariate_columns is not None else None

    # 1. Quantile-transform exposure variables
    transformer = QuantileTransformer(n_quantiles=q, output_distribution='uniform', random_state=42)
    X_exposure = transformer.fit_transform(df[exposure_columns])
    X_exposure = pd.DataFrame(X_exposure, columns=exposure_columns)

    # 2. 准备Outcome variable
    y = df[outcome_column].values

    # 3. Train/validation split
    if validation_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X_exposure, y, test_size=validation_size, random_state=42
        )

        if X_cov is not None:
            X_cov_train, X_cov_test, _, _ = train_test_split(
                X_cov, y, test_size=validation_size, random_state=42
            )
        else:
            X_cov_train, X_cov_test = None, None
    else:
        X_train, y_train = X_exposure, y
        X_test, y_test = None, None
        X_cov_train = X_cov
        X_cov_test = None

    # 4. Weight optimization function
    def wqs_objective(weights, X, y, cov, positive=True):
        """Compute loss for given weights"""
        try:
            # Constrain weights to be positive (if needed)
            if positive:
                weights = np.clip(weights, 0, None)

            # Normalize weights
            weights = weights / np.sum(weights)

            # Compute WQS index
            wqs_index = np.dot(X, weights)

            # Combine covariates and WQS index
            if cov is not None:
                X_design = np.column_stack((wqs_index, cov))
            else:
                X_design = wqs_index.reshape(-1, 1)

            # Add intercept term
            X_design = np.concatenate([np.ones((X_design.shape[0], 1)), X_design], axis=1)

            # Fit linear model
            model = LinearRegression(fit_intercept=False).fit(X_design, y)
            r2 = r2_score(y, model.predict(X_design))
            return -r2  # Minimize negative R²
        except:
            return np.inf  # Return large value when errors occur

    # 5. Estimate weights via bootstrap iterations
    weights_list = []
    r2_train_list = []
    r2_val_list = []

    for i in range(b):
        # Bootstrap重采样
        indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train.iloc[indices].values
        y_boot = y_train[indices]
        cov_boot = X_cov_train.iloc[indices].values if X_cov_train is not None else None

        initial_weights = np.ones(len(exposure_columns)) / len(exposure_columns)
        bounds = [(0.001, 1)] * len(exposure_columns) if positive_weights else [(None, None)] * len(exposure_columns)

        # 约束条件: 权重和为1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        result = minimize(wqs_objective, initial_weights,
                          args=(X_boot, y_boot, cov_boot, positive_weights),
                          method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': 1000})

        if result.success:
            weights = result.x
            weights = weights / weights.sum()  # 确保归一化
            weights_list.append(weights)

            # Training performance
            loss = -result.fun
            r2_train_list.append(loss)

            # Validation performance
            if X_test is not None:
                wqs_index_test = np.dot(X_test.values, weights)
                if X_cov_test is not None:
                    X_test_design = np.column_stack((wqs_index_test, X_cov_test))
                else:
                    X_test_design = wqs_index_test.reshape(-1, 1)

                X_test_design = np.concatenate([np.ones((X_test_design.shape[0], 1)), X_test_design], axis=1)
                y_pred = LinearRegression(fit_intercept=False).fit(X_test_design, y_test).predict(X_test_design)
                r2_val = r2_score(y_test, y_pred)
                r2_val_list.append(r2_val)

    # 6. Compute final weights and performance metrics
    all_weights = np.array(weights_list)
    mean_weights = all_weights.mean(axis=0)
    std_weights = all_weights.std(axis=0)
    weights_df = pd.DataFrame({
        'Variable': exposure_columns,
        'Weight': mean_weights,
        'Std': std_weights,
        'CI_lower': np.quantile(all_weights, 0.025, axis=0),
        'CI_upper': np.quantile(all_weights, 0.975, axis=0)
    }).sort_values('Weight', ascending=False)

    # Overall model performance
    wqs_index_full = np.dot(X_exposure.values, mean_weights)
    if covariate_columns and X_cov is not None:
        X_full_design = np.column_stack((wqs_index_full, X_cov.values))
    else:
        X_full_design = wqs_index_full.reshape(-1, 1)

    X_full_design = np.concatenate([np.ones((X_full_design.shape[0], 1)), X_full_design], axis=1)
    full_model = LinearRegression(fit_intercept=False).fit(X_full_design, y)
    r2_full = r2_score(y, full_model.predict(X_full_design))
    # Add quadratic term of WQS index to assess nonlinearity
    X_full_quad = np.column_stack((wqs_index_full, wqs_index_full ** 2))

    # Include covariates when available
    if covariate_columns and X_cov is not None:
        X_full_design_quad = np.column_stack((X_full_quad, X_cov.values))
    else:
        X_full_design_quad = X_full_quad

    # Add intercept term
    X_full_design_quad = sm.add_constant(X_full_design_quad)

    # Estimate coefficients and p-values using statsmodels
    model_quad = sm.OLS(y, X_full_design_quad).fit()

    # Report coefficients and p-values for mixed exposure (linear term) and quadratic term
    linear_coef = model_quad.params[1]
    linear_pval = model_quad.pvalues[1]

    quadratic_coef = model_quad.params[2]
    quadratic_pval = model_quad.pvalues[2]

    print("\nMixed Exposure and Quadratic Term Analysis:")
    print("==========================================")
    print(f"Mixed Exposure (Linear term): β = {linear_coef:.4f}, p-value = {linear_pval:.4e}")
    print(f"Quadratic term: β = {quadratic_coef:.4f}, p-value = {quadratic_pval:.4e}")


    # 7. Prepare return object
    results = {
        'weights': weights_df,
        'r2_train': np.mean(r2_train_list) if r2_train_list else None,
        'r2_val': np.mean(r2_val_list) if r2_val_list else None,
        'r2_full': r2_full,
        'wqs_index': wqs_index_full,
        'linear_coef': linear_coef,
        'linear_pval': linear_pval,
        'quadratic_coef': quadratic_coef,
        'quadratic_pval': quadratic_pval
    }

    return results


# ===================== Main program =====================
if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    # 1. Read Excel data
    print("Reading data...")
    try:
        # Try reading Excel file
        exposure_df = pd.read_excel('Exposure_Results.xlsx')
        print("Data loaded successfully! Sample size:", len(exposure_df))
    except Exception as e:
        print("Error reading file:", e)
        exit()

    # 2. Define analysis variables (updated exposure pathways)
    exposure_columns = ['soil_ingestion', 'soil_dermal', 'water_ingestion',
                        'water_dermal',
                        'diet_rice', 'diet_solanaceous',
                        'diet_root_tuber', 'diet_legumes', 'diet_other']  # replace diet_total with detailed dietary pathways

    outcome_column = 'Log_Urine_Cd'
    covariate_columns = ['BMI', 'age', 'gender','Smoke','Edu']  # Covariates

    # Check that required columns exist
    required_columns = exposure_columns + [outcome_column] + covariate_columns
    missing_cols = [col for col in required_columns if col not in exposure_df.columns]

    if missing_cols:
        print("Missing columns in data:", missing_cols)
        print("Available columns:", exposure_df.columns.tolist())
        exit()

    # 3. Run WQS analysis
    print("Run WQS analysis ...")
    results = wqs_regression(
        exposure_df,
        exposure_columns,
        outcome_column,
        covariate_columns=covariate_columns,
        q=10,  # 10分位
        b=1000,  # 100次bootstrap
        validation_size=0.2,  # 40%验证集
        positive_weights=True
    )

    # 4. 结果展示
    print("\nWQS analysis results:")
    print("==================================")
    print("Model performance metrics:")
    print(f"- Mean R² (train): {results['r2_train']:.4f}")
    print(f"- Mean R² (validation): {results['r2_val']:.4f}")
    print(f"- R² (full dataset): {results['r2_full']:.4f}")

    print("\nExposure pathway weights (descending by weight):")
    print(results['weights'])

    # 5. Append results back to the dataset
    exposure_df['WQS_index'] = results['wqs_index']

    # 6. 创建专业期刊级可视化
    print("\nCreating publication-quality figures...")

    # Weights plot (Figure 1)
    create_journal_quality_plot(
        data=results['weights'],
        file_name="Fig1_WQS_Weights",
        plot_type="weights"
    )

    # WQS index vs. urinary Cd (Figure 2)
    scatter_data = exposure_df[['WQS_index', outcome_column]].copy()
    # Random subsample to reduce overplotting (keep 50%)
    scatter_data = scatter_data.sample(frac=0.5, random_state=42)
    create_journal_quality_plot(
        data=scatter_data,
        file_name="Fig2_WQS_vs_Urinary_Cd",
        plot_type="scatter",
        outcome_column=outcome_column,
        r2_value=results['r2_full']
    )

    # 7. 保存结果到Excel
    print("\nSaving analysis results to Excel...")
    with pd.ExcelWriter('WQS_Analysis_Results.xlsx') as writer:
        # Exposure Weights表
        results['weights'].to_excel(writer, sheet_name='Exposure Weights', index=False)

        # Full Results
        exposure_df.to_excel(writer, sheet_name='Full Results', index=False)

        # 将这些Metric保存至Model Summary中
        summary_df = pd.DataFrame({
            'Metric': ['R² (train)', 'R² (validation)', 'R² (full dataset)',
                   'Mixed Exposure β', 'Mixed Exposure p-value',
                   'Quadratic β', 'Quadratic p-value',
                   'Bootstrap iterations', 'Number of quantiles'],
            'Value': [results['r2_train'], results['r2_val'], results['r2_full'],
                  results['linear_coef'], results['linear_pval'],
                  results['quadratic_coef'], results['quadratic_pval'],
                  100, 10]
        })
        summary_df.to_excel(writer, sheet_name='Model Summary', index=False)

   
    print("\nWQS analysis finished! Results saved to:")
    print("- Fig1_WQS_Weights.tiff/pdf: Exposure Weights")
    print("- Fig2_WQS_vs_Urinary_Cd.tiff/pdf: WQS index vs. urinary Cd plot")
    print("- WQS_Analysis_Results.xlsx: Complete analysis results")