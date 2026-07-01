%% ========================================================================
% XGBoost for Soil Thermal Diffusivity Prediction
%
% This code accompanies the paper:
% "A Hybrid Machine Learning–Physics Approach for Retrieving Thermal Diffusivity, 
% Simulating Soil Temperature, and Zoning Thermal Regimes in Iran"
%
% GitHub: https://github.com/86311972
% ========================================================================

%% Clean environment
clear; close all; clc;

%% Set random seed for reproducible results
rng(42);

%% Setup paths (USER MUST MODIFY THESE PATHS)
% ============================================================
% PLEASE MODIFY THE PATHS BELOW ACCORDING TO YOUR SYSTEM
% ============================================================

% Path to folder containing raster files (GeoTIFF) for regional prediction
input_data_path = '';  % e.g., 'C:\YourData\rasters\'

% Path to Excel file containing training data
training_data_file = '';  % e.g., 'C:\YourData\station_info_train.xlsx'

% Output directory for results
output_path = '';  % e.g., 'C:\YourResults\XGBoost\'

% Create output directory if it doesn't exist
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

fprintf('Input raster path: %s\n', input_data_path);
fprintf('Training data file: %s\n', training_data_file);
fprintf('Output directory: %s\n', output_path);

%% 1. Load and preprocess training data
fprintf('\n=== 1. Loading Training Data ===\n');

station_data = readtable(training_data_file);

% Extract variables (without LST and DEM)
X_train_original = table2array(station_data(:, {'ndvi', 'sstad', 'sstay', 'bd', ...
                     'clay', 'sand', 'silt', 'soc', 'wv'}));
y_train = station_data.alfa;

% Remove invalid values
valid_samples = ~any(isnan(X_train_original),2) & ~isnan(y_train);
X_train_original = X_train_original(valid_samples,:);
y_train = y_train(valid_samples);

fprintf('Number of training samples after cleaning: %d\n', length(y_train));

%% 2. Soil texture calculation and feature engineering
fprintf('\n=== 2. Feature Engineering ===\n');

clay = X_train_original(:,5);
sand = X_train_original(:,6);
silt = X_train_original(:,7);
total = clay + sand + silt;

% Calculate soil texture index (weighted average)
soil_texture_index = (clay./total)*0.4 + (sand./total)*0.3 + (silt./total)*0.3;

% Create new feature matrix
X_train_new = [X_train_original(:,1:4), X_train_original(:,8:9), soil_texture_index];
variable_names_cell = {'ndvi', 'sstad', 'sstay', 'bd', 'soc', 'wv', 'soil_texture'};
fprintf('Number of features after engineering: %d\n', size(X_train_new, 2));

%% 3. Min-Max Normalization
fprintf('\n=== 3. Data Normalization ===\n');

[X_train_norm, x_min, x_range] = normalize_minmax(X_train_new);
fprintf('Data normalized to [0, 1] range\n');

%% 4. Multicollinearity analysis and VIF calculation
fprintf('\n=== 4. Multicollinearity Analysis (VIF) ===\n');

correlation_matrix = corr(X_train_norm);
num_vars = size(X_train_norm,2);
vif_values = zeros(1,num_vars);

for i=1:num_vars
    other_vars = [1:i-1, i+1:num_vars];
    X_other = X_train_norm(:,other_vars);
    X_target = X_train_norm(:,i);
    beta = regress(X_target, [ones(size(X_other,1),1), X_other]);
    predicted = [ones(size(X_other,1),1), X_other]*beta;
    r_squared = 1 - sum((X_target - predicted).^2)/sum((X_target - mean(X_target)).^2);
    vif_values(i) = 1/(1-r_squared);
end

% Display VIF results
fprintf('\nVariance Inflation Factor (VIF) results:\n');
fprintf('----------------------------------------\n');
for i = 1:num_vars
    if vif_values(i) > 10
        status = 'Severe multicollinearity';
    elseif vif_values(i) > 5
        status = 'Moderate multicollinearity';
    else
        status = 'Acceptable';
    end
    fprintf('%s: VIF = %.2f (%s)\n', variable_names_cell{i}, vif_values(i), status);
end

%% 5. Configure Python and import XGBoost
fprintf('\n=== 5. Configuring Python and Loading XGBoost ===\n');

% Python path (USER MUST MODIFY THIS PATH TO THEIR PYTHON INSTALLATION)
% Common Python paths:
% - Windows: 'C:\Users\username\AppData\Local\Programs\Python\Python311\python.exe'
% - Linux/Mac: '/usr/bin/python3' or '/usr/local/bin/python3'
python_path = '';  % e.g., 'C:\Users\username\AppData\Local\Programs\Python\Python311\python.exe'

try
    pyenv('Version', python_path);
    fprintf('Python set to: %s\n', python_path);
catch
    error('Python not found. Please set the correct Python path in the code.');
end

% Import XGBoost and NumPy
xgb = py.importlib.import_module('xgboost');
np = py.importlib.import_module('numpy');
fprintf('XGBoost and NumPy loaded successfully.\n');

%% 6. Data split for final evaluation (80-20)
fprintf('\n=== 6. Data Split (80%% Training, 20%% Validation) ===\n');

cv_holdout = cvpartition(length(y_train), 'HoldOut', 0.2);
idxTrain_hold = training(cv_holdout);
idxVal_hold = test(cv_holdout);

X_train_hold = X_train_norm(idxTrain_hold, :);
y_train_hold = y_train(idxTrain_hold);
X_val_hold = X_train_norm(idxVal_hold, :);
y_val_hold = y_train(idxVal_hold);

fprintf('Training samples: %d, Validation samples: %d\n', sum(idxTrain_hold), sum(idxVal_hold));

%% 7. Hyperparameter optimization with cross-validation on full dataset
fprintf('\n=== 7. Hyperparameter Optimization (5-fold CV on Full Dataset) ===\n');

param_grid = struct();
param_grid.max_depth = [2, 3, 4];
param_grid.eta = [0.01, 0.02, 0.05];
param_grid.num_round = [80, 100];
param_grid.subsample = [0.5, 0.6];
param_grid.colsample_bytree = [0.5, 0.6];
param_grid.reg_alpha = [2.0, 2.5];
param_grid.reg_lambda = [3.0, 4.0, 5.0];
param_grid.min_child_weight = [10, 20];
param_grid.gamma = [0.2, 0.3, 0.4];

results = [];
best_r2_cv = -Inf;
best_params = struct();
best_validation_metrics = struct();

max_combinations_to_test = 50;
fprintf('Total combinations to test: %d\n', max_combinations_to_test);

for i = 1:max_combinations_to_test
    md = param_grid.max_depth(randi(length(param_grid.max_depth)));
    eta = param_grid.eta(randi(length(param_grid.eta)));
    n_round = param_grid.num_round(randi(length(param_grid.num_round)));
    subsample = param_grid.subsample(randi(length(param_grid.subsample)));
    colsample = param_grid.colsample_bytree(randi(length(param_grid.colsample_bytree)));
    reg_alpha = param_grid.reg_alpha(randi(length(param_grid.reg_alpha)));
    reg_lambda = param_grid.reg_lambda(randi(length(param_grid.reg_lambda)));
    min_child = param_grid.min_child_weight(randi(length(param_grid.min_child_weight)));
    gamma_val = param_grid.gamma(randi(length(param_grid.gamma)));
    
    fprintf('Processing combination %d: max_depth=%d, eta=%.3f, rounds=%d\n', i, md, eta, n_round);
    
    k = 5;
    cv = cvpartition(length(y_train), 'KFold', k);
    cv_r2 = zeros(k, 1);
    cv_rmse = zeros(k, 1);
    cv_mae = zeros(k, 1);
    
    for fold = 1:k
        idxTrain = training(cv, fold);
        idxTest = test(cv, fold);
        
        X_train_fold = X_train_norm(idxTrain,:);
        y_train_fold = y_train(idxTrain);
        X_test_fold = X_train_norm(idxTest,:);
        y_test_fold = y_train(idxTest);
        
        X_train_py_fold = py.numpy.array(X_train_fold);
        y_train_py_fold = py.numpy.array(y_train_fold);
        X_test_py_fold = py.numpy.array(X_test_fold);
        
        dtrain_cv = xgb.DMatrix(X_train_py_fold, y_train_py_fold);
        dtest_cv = xgb.DMatrix(X_test_py_fold);
        
        params = py.dict(pyargs(...
            'max_depth', int32(md), ...
            'eta', double(eta), ...
            'subsample', double(subsample), ...
            'colsample_bytree', double(colsample), ...
            'reg_alpha', double(reg_alpha), ...
            'reg_lambda', double(reg_lambda), ...
            'min_child_weight', int32(min_child), ...
            'gamma', double(gamma_val), ...
            'objective', 'reg:squarederror', ...
            'random_state', int32(42), ...
            'verbosity', int32(0)));
        
        bst = xgb.train(params, dtrain_cv, int32(n_round));
        
        y_pred_py = bst.predict(dtest_cv);
        y_pred_cv = double(py.array.array('d', py.numpy.nditer(y_pred_py)));
        y_pred_cv = y_pred_cv(:);
        
        cv_r2(fold) = corr(y_pred_cv, y_test_fold)^2;
        residuals = y_pred_cv - y_test_fold;
        cv_rmse(fold) = sqrt(mean(residuals.^2));
        cv_mae(fold) = mean(abs(residuals));
    end
    
    mean_r2 = mean(cv_r2);
    mean_rmse = mean(cv_rmse);
    mean_mae = mean(cv_mae);
    std_r2 = std(cv_r2);
    std_rmse = std(cv_rmse);
    std_mae = std(cv_mae);
    
    result = struct();
    result.MaxDepth = md;
    result.Eta = eta;
    result.NumRound = n_round;
    result.Subsample = subsample;
    result.ColsampleByTree = colsample;
    result.RegAlpha = reg_alpha;
    result.RegLambda = reg_lambda;
    result.MinChildWeight = min_child;
    result.Gamma = gamma_val;
    result.Mean_R2 = mean_r2;
    result.Std_R2 = std_r2;
    result.Mean_RMSE = mean_rmse;
    result.Std_RMSE = std_rmse;
    result.Mean_MAE = mean_mae;
    result.Std_MAE = std_mae;
    
    results = [results; result];
    
    if mean_r2 > best_r2_cv
        best_r2_cv = mean_r2;
        best_params.max_depth = md;
        best_params.eta = eta;
        best_params.num_round = n_round;
        best_params.subsample = subsample;
        best_params.colsample_bytree = colsample;
        best_params.reg_alpha = reg_alpha;
        best_params.reg_lambda = reg_lambda;
        best_params.min_child_weight = min_child;
        best_params.gamma = gamma_val;
        best_validation_metrics.Mean_R2 = mean_r2;
        best_validation_metrics.Std_R2 = std_r2;
        best_validation_metrics.Mean_RMSE = mean_rmse;
        best_validation_metrics.Std_RMSE = std_rmse;
        best_validation_metrics.Mean_MAE = mean_mae;
        best_validation_metrics.Std_MAE = std_mae;
    end
end

fprintf('Best cross-validated R²: %.4f\n', best_r2_cv);

%% 7.5 Display optimal hyperparameters
fprintf('\n========================================\n');
fprintf('      Optimal XGBoost Hyperparameters     \n');
fprintf('========================================\n');
fprintf('Number of rounds (num_round): %d\n', best_params.num_round);
fprintf('Max depth: %d\n', best_params.max_depth);
fprintf('Learning rate (eta): %.4f\n', best_params.eta);
fprintf('Subsample ratio: %.2f\n', best_params.subsample);
fprintf('Colsample by tree: %.2f\n', best_params.colsample_bytree);
fprintf('L1 regularization (alpha): %.2f\n', best_params.reg_alpha);
fprintf('L2 regularization (lambda): %.2f\n', best_params.reg_lambda);
fprintf('Min Child Weight: %d\n', best_params.min_child_weight);
fprintf('Gamma: %.2f\n', best_params.gamma);
fprintf('Cross-validated R²: %.4f ± %.4f\n', ...
    best_validation_metrics.Mean_R2, best_validation_metrics.Std_R2);
fprintf('Cross-validated RMSE: %.4f\n', best_validation_metrics.Mean_RMSE);
fprintf('Cross-validated MAE: %.4f\n', best_validation_metrics.Mean_MAE);
fprintf('========================================\n');

%% 8. Train final model on full dataset (337 samples)
fprintf('\n=== 8. Training Final Model on Full Dataset (337 samples) ===\n');

X_train_py_full = py.numpy.array(X_train_norm);
y_train_py_full = py.numpy.array(y_train);
dtrain_full = xgb.DMatrix(X_train_py_full, y_train_py_full);

params_final = py.dict(pyargs(...
    'max_depth', int32(best_params.max_depth), ...
    'eta', double(best_params.eta), ...
    'subsample', double(best_params.subsample), ...
    'colsample_bytree', double(best_params.colsample_bytree), ...
    'reg_alpha', double(best_params.reg_alpha), ...
    'reg_lambda', double(best_params.reg_lambda), ...
    'min_child_weight', int32(best_params.min_child_weight), ...
    'gamma', double(best_params.gamma), ...
    'objective', 'reg:squarederror', ...
    'random_state', int32(42), ...
    'verbosity', int32(1)));

bst_final = xgb.train(params_final, dtrain_full, int32(best_params.num_round));

%% 9. Evaluation on full dataset (training)
fprintf('\n=== 9. Training Data Evaluation ===\n');

y_pred_train_py = bst_final.predict(dtrain_full);
y_pred_train = double(py.array.array('d', py.numpy.nditer(y_pred_train_py)));
y_pred_train = y_pred_train(:);

train_r2 = corr(y_pred_train, y_train)^2;
train_rmse = sqrt(mean((y_pred_train - y_train).^2));
train_mae = mean(abs(y_pred_train - y_train));
train_mse = mean((y_pred_train - y_train).^2);
train_nse = 1 - sum((y_train - y_pred_train).^2) / sum((y_train - mean(y_train)).^2);
train_pbias = 100 * sum(y_train - y_pred_train) / sum(y_train);
train_residuals = y_pred_train - y_train;

%% 10. Evaluation on validation data (20% hold-out)
fprintf('\n=== 10. Validation Data Evaluation ===\n');

X_val_py = py.numpy.array(X_val_hold);
dval = xgb.DMatrix(X_val_py);

y_pred_val_py = bst_final.predict(dval);
y_pred_val = double(py.array.array('d', py.numpy.nditer(y_pred_val_py)));
y_pred_val = y_pred_val(:);

val_r2 = corr(y_pred_val, y_val_hold)^2;
val_rmse = sqrt(mean((y_pred_val - y_val_hold).^2));
val_mae = mean(abs(y_pred_val - y_val_hold));
val_mse = mean((y_pred_val - y_val_hold).^2);
val_nse = 1 - sum((y_val_hold - y_pred_val).^2) / sum((y_val_hold - mean(y_val_hold)).^2);
val_pbias = 100 * sum(y_val_hold - y_pred_val) / sum(y_val_hold);
val_residuals = y_pred_val - y_val_hold;

%% 11. Feature importance calculation (on full dataset)
fprintf('\n=== 11. Feature Importance (Full Dataset) ===\n');

try
    importance_dict = bst_final.get_score(pyargs('importance_type', 'gain'));
    
    keys_py = py.list(importance_dict.keys());
    values_py = py.list(importance_dict.values());
    
    keys_cell = cell(keys_py);
    values_cell = cell(values_py);
    
    importance_values = zeros(1, length(keys_cell));
    for i = 1:length(keys_cell)
        importance_values(i) = double(values_cell{i});
    end
    
    importance_raw = zeros(1, length(variable_names_cell));
    for i = 1:length(keys_cell)
        key_str = char(keys_cell{i});
        idx = str2double(key_str(2:end)) + 1;
        if idx <= length(importance_raw)
            importance_raw(idx) = importance_values(i);
        end
    end
    
    relative_importance_xgb = importance_raw / sum(importance_raw);
    
catch ME
    fprintf('Error in importance calculation: %s\n', ME.message);
    importance_raw = ones(1, length(variable_names_cell));
    relative_importance_xgb = importance_raw / sum(importance_raw);
end

%% 12. Final performance summary table
fprintf('\n================================================================================\n');
fprintf('                     XGBoost Model Performance Summary                          \n');
fprintf('================================================================================\n');
fprintf('Metric    |  Training   | Validation  |  CV (Mean ± Std)\n');
fprintf('--------------------------------------------------------------------------------\n');
fprintf('R²        |   %.4f    |   %.4f    |   %.4f ± %.4f\n', ...
    train_r2, val_r2, best_validation_metrics.Mean_R2, best_validation_metrics.Std_R2);
fprintf('RMSE      |   %.4f    |   %.4f    |   %.4f ± %.4f\n', ...
    train_rmse, val_rmse, best_validation_metrics.Mean_RMSE, best_validation_metrics.Std_RMSE);
fprintf('MAE       |   %.4f    |   %.4f    |   %.4f ± %.4f\n', ...
    train_mae, val_mae, best_validation_metrics.Mean_MAE, best_validation_metrics.Std_MAE);
fprintf('MSE       |   %.4f    |   %.4f    |       -\n', train_mse, val_mse);
fprintf('NSE       |   %.4f    |   %.4f    |       -\n', train_nse, val_nse);
fprintf('PBIAS     |   %.2f%%   |   %.2f%%   |       -\n', train_pbias, val_pbias);
fprintf('================================================================================\n');

% Save results table as CSV
results_table = table(...
    [train_r2; train_rmse; train_mae; train_mse; train_nse; train_pbias], ...
    [val_r2; val_rmse; val_mae; val_mse; val_nse; val_pbias], ...
    [best_validation_metrics.Mean_R2; best_validation_metrics.Mean_RMSE; best_validation_metrics.Mean_MAE; NaN; NaN; NaN], ...
    'VariableNames', {'Training', 'Validation', 'CV'}, ...
    'RowNames', {'R²', 'RMSE', 'MAE', 'MSE', 'NSE', 'PBIAS'});
writetable(results_table, fullfile(output_path, 'XGBoost_Results.csv'), 'WriteRowNames', true);

%% 13. Display variable importance
fprintf('\n=== Relative Variable Importance (Full Dataset - 337 samples) ===\n');
fprintf('-------------------------------------------------------------\n');
for i = 1:length(variable_names_cell)
    fprintf('%s: %.4f\n', variable_names_cell{i}, relative_importance_xgb(i));
end

% Save importance table
importance_table = table(variable_names_cell', relative_importance_xgb', ...
    'VariableNames', {'Variable', 'Relative_Importance'});
writetable(importance_table, fullfile(output_path, 'XGBoost_Importance.csv'));

%% 14. Generate publication-quality figures
fprintf('\n=== 14. Generating Figures ===\n');

figure('Position', [100, 100, 1400, 1000], 'Color', 'white');

% Figure 1: Correlation matrix
subplot(2,3,1);
imagesc(correlation_matrix);
colorbar;
colormap(jet);
set(gca, 'XTick', 1:length(variable_names_cell), ...
         'YTick', 1:length(variable_names_cell), ...
         'XTickLabel', variable_names_cell, ...
         'YTickLabel', variable_names_cell);
xtickangle(45);
title('(a) Correlation Matrix', 'FontSize', 12, 'FontWeight', 'bold');

% Figure 2: VIF values
subplot(2,3,2);
bar(vif_values, 'FaceColor', [0.2 0.6 0.8]);
title('(b) Variance Inflation Factor (VIF)', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTickLabel', variable_names_cell);
xtickangle(45);
ylabel('VIF Value');
grid on;
hold on;
plot(xlim, [5, 5], 'r--', 'LineWidth', 2);
plot(xlim, [10, 10], 'r:', 'LineWidth', 2);
legend({'VIF', 'Threshold (VIF=5)', 'Threshold (VIF=10)'}, 'Location', 'northeast');

% Figure 3: Variable importance
subplot(2,3,3);
bar(relative_importance_xgb, 'FaceColor', [0.8 0.4 0.2]);
title('(c) Variable Importance', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTickLabel', variable_names_cell);
xtickangle(45);
ylabel('Relative Importance');
grid on;

% Figure 4: Predicted vs Observed (Training)
subplot(2,3,4);
scatter(y_train, y_pred_train, 50, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--', 'LineWidth', 2);
xlabel('Observed Values');
ylabel('Predicted Values');
title(sprintf('(d) Training: R² = %.3f', train_r2), 'FontSize', 12);
grid on;

% Figure 5: Residual distribution
subplot(2,3,5);
histogram(train_residuals, 20, 'FaceColor', [0.6 0.3 0.7], 'FaceAlpha', 0.7);
xlabel('Residuals');
ylabel('Frequency');
title('(e) Residual Distribution', 'FontSize', 12);
grid on;

% Figure 6: Performance comparison
subplot(2,3,6);
metrics = [train_r2, val_r2; train_rmse, val_rmse; train_mae, val_mae]';
bar(metrics);
set(gca, 'XTickLabel', {'R²', 'RMSE', 'MAE'});
ylabel('Value');
title('(f) Training vs Validation Performance', 'FontSize', 12);
legend('Training (Full)', 'Validation (20%)', 'Location', 'best');
grid on;

% Save figure
print(gcf, fullfile(output_path, 'XGBoost_Results.png'), '-dpng', '-r300');
fprintf('Figure saved: %s\n', fullfile(output_path, 'XGBoost_Results.png'));

%% 15. Save results
fprintf('\n=== 15. Saving Results ===\n');

save(fullfile(output_path, 'XGBoost_Model.mat'), ...
     'bst_final', 'best_params', 'best_validation_metrics', ...
     'train_r2', 'val_r2', 'train_rmse', 'val_rmse', ...
     'train_mae', 'val_mae', 'vif_values', 'correlation_matrix', ...
     'x_min', 'x_range', 'relative_importance_xgb', 'results', '-v7.3');

% Save report
report_file = fullfile(output_path, 'XGBoost_Report.txt');
fid = fopen(report_file, 'w');

fprintf(fid, '========================================\n');
fprintf(fid, 'XGBoost Model Summary Report\n');
fprintf(fid, '========================================\n\n');

fprintf(fid, 'Optimal Hyperparameters:\n');
fprintf(fid, '  Max depth: %d\n', best_params.max_depth);
fprintf(fid, '  Learning rate (eta): %.4f\n', best_params.eta);
fprintf(fid, '  Number of rounds: %d\n', best_params.num_round);
fprintf(fid, '  Subsample ratio: %.1f\n', best_params.subsample);
fprintf(fid, '  Colsample by tree: %.1f\n', best_params.colsample_bytree);
fprintf(fid, '  L1 regularization (alpha): %.1f\n', best_params.reg_alpha);
fprintf(fid, '  L2 regularization (lambda): %.1f\n', best_params.reg_lambda);
fprintf(fid, '  Min Child Weight: %d\n', best_params.min_child_weight);
fprintf(fid, '  Gamma: %.1f\n\n', best_params.gamma);

fprintf(fid, 'Performance Summary:\n');
fprintf(fid, '--------------------------------------------------------------------------------\n');
fprintf(fid, 'Metric    |  Training   | Validation  |  CV (Mean ± Std)\n');
fprintf(fid, '--------------------------------------------------------------------------------\n');
fprintf(fid, 'R²        |   %.4f    |   %.4f    |   %.4f ± %.4f\n', ...
    train_r2, val_r2, best_validation_metrics.Mean_R2, best_validation_metrics.Std_R2);
fprintf(fid, 'RMSE      |   %.4f    |   %.4f    |   %.4f ± %.4f\n', ...
    train_rmse, val_rmse, best_validation_metrics.Mean_RMSE, best_validation_metrics.Std_RMSE);
fprintf(fid, 'MAE       |   %.4f    |   %.4f    |   %.4f ± %.4f\n', ...
    train_mae, val_mae, best_validation_metrics.Mean_MAE, best_validation_metrics.Std_MAE);
fprintf(fid, 'MSE       |   %.4f    |   %.4f    |       -\n', train_mse, val_mse);
fprintf(fid, 'NSE       |   %.4f    |   %.4f    |       -\n', train_nse, val_nse);
fprintf(fid, 'PBIAS     |   %.2f%%   |   %.2f%%   |       -\n', train_pbias, val_pbias);
fprintf(fid, '--------------------------------------------------------------------------------\n\n');

fprintf(fid, 'Variable Importance:\n');
for i = 1:length(variable_names_cell)
    fprintf(fid, '  %s: %.4f\n', variable_names_cell{i}, relative_importance_xgb(i));
end

fprintf(fid, '\nVIF Analysis:\n');
for i = 1:length(variable_names_cell)
    status = 'Acceptable';
    if vif_values(i) > 10
        status = 'Severe multicollinearity';
    elseif vif_values(i) > 5
        status = 'Moderate multicollinearity';
    end
    fprintf(fid, '  %s: %.2f (%s)\n', variable_names_cell{i}, vif_values(i), status);
end

fclose(fid);

%% 16. Regional prediction (GeoTIFF raster files)
fprintf('\n=== 16. Regional Prediction ===\n');

% List of raster files (USER MUST MATCH THESE NAMES TO THEIR FILES)
raster_names = {'ndvi_yearly', 'ssta_day', 'ssta_yearly', ...
                'bd_mean_50cm', 'clay_mean_50cm', 'sand_mean_50cm', ...
                'silt_mean_50cm', 'soc_mean_50cm', 'wv_mean_50cm'};

fprintf('Loading raster files...\n');

% Read first raster to get reference info
first_raster = fullfile(input_data_path, [raster_names{1}, '.tif']);
if ~exist(first_raster, 'file')
    error('Raster file not found: %s\nPlease check input_data_path', first_raster);
end

[ref_matrix, R] = readgeoraster(first_raster);
num_rows = size(ref_matrix,1);
num_cols = size(ref_matrix,2);
num_vars = length(raster_names);

% Load all rasters
all_layers = zeros(num_rows, num_cols, num_vars, 'single');
for i = 1:num_vars
    filename = fullfile(input_data_path, [raster_names{i}, '.tif']);
    [layer, ~] = readgeoraster(filename);
    all_layers(:,:,i) = single(layer);
    fprintf('  Loaded: %s\n', raster_names{i});
end

% Calculate soil texture for the region
fprintf('Calculating soil texture index for region...\n');
clay_map = all_layers(:,:,5);
sand_map = all_layers(:,:,6);
silt_map = all_layers(:,:,7);

total_map = clay_map + sand_map + silt_map;
total_map(total_map == 0) = 1;

clay_percent_map = (clay_map ./ total_map) * 100;
sand_percent_map = (sand_map ./ total_map) * 100;
silt_percent_map = (silt_map ./ total_map) * 100;

soil_texture_map = (clay_percent_map * 0.4 + sand_percent_map * 0.3 + silt_percent_map * 0.3);

% Create feature matrix for region
X_predict_original = reshape(all_layers, [], num_vars);
X_predict_new = [X_predict_original(:,1:4), X_predict_original(:,8:9), reshape(soil_texture_map, [], 1)];

% Identify valid pixels
valid_pixels = ~any(isnan(X_predict_new),2) & ~any(isinf(X_predict_new),2);
fprintf('Valid pixels for prediction: %d out of %d\n', sum(valid_pixels), length(valid_pixels));

% Normalize region data using training parameters
X_predict_norm = zeros(size(X_predict_new), 'like', X_predict_new);
X_predict_norm(valid_pixels,:) = (X_predict_new(valid_pixels,:) - x_min) ./ x_range;
X_predict_norm(~isfinite(X_predict_norm)) = 0;

% Predict using XGBoost model
fprintf('Predicting thermal diffusivity for region...\n');
tic;
dX_predict = xgb.DMatrix(py.numpy.array(X_predict_norm(valid_pixels,:)));

y_pred_region_py = bst_final.predict(dX_predict);
y_pred_region = double(py.array.array('d', py.numpy.nditer(y_pred_region_py)));
y_pred_region = y_pred_region(:);
toc;

% Reconstruct full map
thermal_cond_map = NaN(num_rows, num_cols, 'single');
thermal_cond_map(valid_pixels) = y_pred_region;

% Save predicted map
output_raster = fullfile(output_path, 'Thermal_Conductivity_XGB.tif');
geotiffwrite(output_raster, thermal_cond_map, R, 'CoordRefSysCode', 4326);
fprintf('Regional prediction map saved: %s\n', output_raster);

% Create color map figure
figure('Position', [200, 200, 800, 600], 'Color', 'white');
imagesc(thermal_cond_map, 'AlphaData', ~isnan(thermal_cond_map));
colorbar;
colormap(jet);
title('Predicted Soil Thermal Diffusivity Map (XGBoost)', 'FontSize', 14);
xlabel('Column');
ylabel('Row');
axis equal;
print(gcf, fullfile(output_path, 'Thermal_Conductivity_Map_XGB.png'), '-dpng', '-r300');

fprintf('\n========================================\n');
fprintf('Process completed successfully!\n');
fprintf('Results saved in: %s\n', output_path);
fprintf('========================================\n');

%% ========================================================================
% Helper Functions
% ========================================================================

function [data_norm, min_vals, range_vals] = normalize_minmax(data)
    % Min-Max normalization to [0, 1] range
    min_vals = min(data,[],1);
    max_vals = max(data,[],1);
    range_vals = max_vals - min_vals;
    range_vals(range_vals==0) = 1;
    data_norm = (data - min_vals) ./ range_vals;
end
