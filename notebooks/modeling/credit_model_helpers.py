import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


def aggregate_numeric(df, group_col, agg_col, operations):
    agg_dict = {f'{agg_col}_{op}': (agg_col, op) for op in operations}
    return df.groupby(group_col).agg(**agg_dict).reset_index()


def aggregate_categorical_ohe(df, group_col, cat_col, prefix):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[[cat_col]])
    
    encoded_cols = [f'{prefix}_{cat}' for cat in encoder.categories_[0]]
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols)
    encoded_df[group_col] = df[group_col].values
    
    return encoded_df.groupby(group_col).sum().reset_index()


def filter_by_missing(df, threshold=0.80):
    missing_pct = df.isnull().sum() / len(df)
    to_drop = missing_pct[missing_pct > threshold].index.tolist()
    return df.drop(columns=to_drop), to_drop


def remove_low_variance(df, threshold=0.01):
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df)
    cols_to_keep = df.columns[selector.get_support()].tolist()
    cols_dropped = [col for col in df.columns if col not in cols_to_keep]
    return df[cols_to_keep], cols_dropped


def remove_correlated(df, threshold=0.95, importances=None):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
    
    to_drop = []
    for col in upper.columns:
        correlated_features = upper.index[upper[col] > threshold].tolist()
        
        if correlated_features:
            if importances is not None:
                imp_dict = dict(zip(importances['feature'], importances['importance']))
                candidates = [col] + correlated_features
                candidates_in_imp = [c for c in candidates if c in imp_dict]
                
                if candidates_in_imp:
                    sorted_candidates = sorted(candidates_in_imp, key=lambda x: imp_dict[x])
                    to_drop.extend(sorted_candidates[:-1])
            else:
                to_drop.extend(correlated_features)
    
    to_drop = list(set(to_drop))
    return df.drop(columns=to_drop), to_drop


def get_feature_importances(model, feature_names):
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importances


def select_by_importance_threshold(importances_df, threshold):
    selected = importances_df[importances_df['importance'] >= threshold]
    return selected['feature'].tolist()


def train_preliminary_model(X_train, y_train, X_val, y_val, model_type='lightgbm'):
    if model_type == 'lightgbm':
        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            random_state=42,
            class_weight='balanced',
            verbose=-1
        )
    else:
        model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            scale_pos_weight=11.4,
            random_state=42,
            tree_method='hist',
            eval_metric='auc',
            verbosity=0
        )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    
    return model, train_auc, val_auc


def compare_feature_sets(X_train_before, X_train_after, y_train, X_val_before, X_val_after, y_val, model_type='lightgbm'):
    print("Training with ALL features...")
    model_before, train_auc_before, val_auc_before = train_preliminary_model(
        X_train_before, y_train, X_val_before, y_val, model_type
    )
    
    print("Training with SELECTED features...")
    model_after, train_auc_after, val_auc_after = train_preliminary_model(
        X_train_after, y_train, X_val_after, y_val, model_type
    )
    
    results = {
        'n_features_before': X_train_before.shape[1],
        'n_features_after': X_train_after.shape[1],
        'n_features_removed': X_train_before.shape[1] - X_train_after.shape[1],
        'train_auc_before': train_auc_before,
        'train_auc_after': train_auc_after,
        'val_auc_before': val_auc_before,
        'val_auc_after': val_auc_after,
        'val_auc_change': val_auc_after - val_auc_before,
        'model_after': model_after
    }
    
    return results


def print_comparison_results(results, model_name='Model'):
    print("\n" + "="*60)
    print(f"{model_name.upper()} - FEATURE SELECTION COMPARISON")
    print("="*60)
    print(f"Features: {results['n_features_before']} → {results['n_features_after']} "
          f"(removed {results['n_features_removed']})")
    print(f"\nBefore Selection:")
    print(f"  Train AUC: {results['train_auc_before']:.4f}")
    print(f"  Val AUC:   {results['val_auc_before']:.4f}")
    print(f"\nAfter Selection:")
    print(f"  Train AUC: {results['train_auc_after']:.4f}")
    print(f"  Val AUC:   {results['val_auc_after']:.4f}")
    print(f"\nValidation AUC Change: {results['val_auc_change']:+.4f}")
    
    if results['val_auc_change'] >= 0:
        print("Feature selection IMPROVED or maintained performance")
    else:
        print("Feature selection DEGRADED performance")
    print("="*60)

