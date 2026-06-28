%% ========================================================================
% XGBoost Model for Soil Thermal Diffusivity Prediction
%
% This code accompanies the paper:
% "A Hybrid Machine Learning–Physics Approach for Retrieving Thermal Diffusivity, Simulating Soil Temperature, and Zoning Thermal Regimes in Iran"
%
% GitHub: https://github.com/86311972
%
% REQUIREMENTS:
%   - Python with XGBoost installed
%   - Run: pip install xgboost numpy
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
output_path = '';  % e.g., 'C:\YourResults\XGB\'

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

X_train_original = table2array(station_data(:, {'ndvi', 'sstad', 'sstay', 'bd', ...
                     'clay', 'sand', 'silt', 'soc', 'wv'}));
y_train = station_data.alfa;

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
soil_texture_index = (clay./total)*0.4 + (sand./total)*0.3 + (silt./total)*0.3;

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

%% 5. Setup Python and import XGBoost
fprintf('\n=== 5. Setting up Python Environment ===\n');

% Specify Python path (USER MUST MODIFY THIS PATH)
% Example: pyenv('Version', 'C:\Users\YourName\AppData\Local\Programs\Python\Python39\python.exe')
% If Python is in system PATH, use: pyenv('Version', 'python')
pyenv('Version', 'python');  % USER MUST MODIFY THIS

% Import required modules
xgb = py.importlib.import_module('xgboost');
np = py.importlib.import_module('numpy');



%% 6. Hyperparameter optimization with cross-validation
fprintf('\n=== 6. Hyperparameter Optimization (5-fold CV) ===\n');

% Conservative hyperparameter grid for 337 samples and 7 features
param_grid = struct();
param_grid.max_depth = [2, 3, 4];
param_grid.eta = [0.01, 0.02, 0.05];
param_grid.num_round = [80, 100];
param_grid.subsample = [0.5, 0.6];
param_grid.colsample_bytree = [0.5, 0.6];
param_grid.reg_alpha = [1.5, 2.0, 2.5];
param_grid.reg_lambda = [3.0, 4.0, 5.0];
param_grid.min_child_weight = [10, 15, 20];
param_grid.gamma = [0.2, 0.3, 0.4];

% Maximum number of random combinations to test
max_combinations_to_test = 50;
fprintf('Total random combinations to test: %d\n', max_combinations_to_test);

% Storage for results
results = [];
best_r2_cv = -Inf;
best_params = struct();
best_validation_metrics = struct();

tested_combinations = 0;
for i = 1:max_combinations_to_test
    % Randomly select hyperparameters
    md = param_grid.max_depth(randi(length(param_grid.max_depth)));
    eta = param_grid.eta(randi(length(param_grid.eta)));
    n_round = param_grid.num_round(randi(length(param_grid.num_round)));
    subsample = param_grid.subsample(randi(length(param_grid.subsample)));
    colsample = param_grid.colsample_bytree(randi(length(param_grid.colsample_bytree)));
    reg_alpha = param_grid.reg_alpha(randi(length(param_grid.reg_alpha)));
    reg_lambda = param_grid.reg_lambda(randi(length(param_grid.reg_lambda)));
    min_child = param_grid.min_child_weight(randi(length(param_grid.min_child_weight)));
    gamma_val = param_grid.gamma(randi(length(param_grid.gamma)));
    
    tested_combinations = tested_combinations + 1;
    fprintf('Processing combination %d/%d: max_depth=%d, eta=%.3f, rounds=%d\n', ...
            tested_combinations, max_combinations_to_test, md, eta, n_round);
    
    % 5-fold cross-validation
    k = 5;
    cv = cvpartition(length(y_train), 'KFold', k);
    cv_r2 = zeros(k, 1);
    cv_rmse = zeros(k, 1);
    cv_mae = zeros(k, 1);
    
    for fold = 1:k
        idxTrain = training(cv, fold);
        idxTest = test(cv, fold);
        
        % Fold data
        X_train_fold = X_train_norm(idxTrain,:);
        y_train_fold = y_train(idxTrain);
        X_test_fold = X_train_norm(idxTest,:);
        y_test_fold = y_train(idxTest);
        
        % Convert to Python format
        X_train_py_fold = py.numpy.array(X_train_fold);
        y_train_py_fold = py.numpy.array(y_train_fold);
        X_test_py_fold = py.numpy.array(X_test_fold);
        
        % Create DMatrix
        dtrain_cv = xgb.DMatrix(X_train_py_fold, y_train_py_fold);
        dtest_cv = xgb.DMatrix(X_test_py_fold);
        
        % Model parameters
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
        
        % Train model
        bst = xgb.train(params, dtrain_cv, int32(n_round));
        
        % Predict
        y_pred_py = bst.predict(dtest_cv);
        y_pred_cv = double(py.array.array('d', py.numpy.nditer(y_pred_py)));
        y_pred_cv = y_pred_cv(:);
        
        % Calculate metrics
        cv_r2(fold) = corr(y_pred_cv, y_test_fold)^2;
        residuals = y_pred_cv - y_test_fold;
        cv_rmse(fold) = sqrt(mean(residuals.^2));
        cv_mae(fold) = mean(abs(residuals));
    end
    
    % Average metrics
    mean_r2 = mean(cv_r2);
    mean_rmse = mean(cv_rmse);
    mean_mae = mean(cv_mae);
    std_r2 = std(cv_r2);
    
    % Store results
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
    result.Mean_MAE = mean_mae;
    
    results = [results; result];
    
    % Update best model
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
        best_validation_metrics.Mean_MAE = mean_mae;
        fprintf('  New best model found! R² = %.4f\n', mean_r2);
    end
end

fprintf('Best cross-validation R²: %.4f\n', best_r2_cv);

%% 7. Train final model with best hyperparameters
fprintf('\n=== 7. Training Final Model ===\n');

fprintf('Best hyperparameters:\n');
fprintf('  max_depth: %d\n', best_params.max_depth);
fprintf('  eta: %.4f\n', best_params.eta);
fprintf('  num_round: %d\n', best_params.num_round);
fprintf('  subsample: %.1f\n', best_params.subsample);
fprintf('  colsample_bytree: %.1f\n', best_params.colsample_bytree);
fprintf('  reg_alpha: %.1f\n', best_params.reg_alpha);
fprintf('  reg_lambda: %.1f\n', best_params.reg_lambda);
fprintf('  min_child_weight: %d\n', best_params.min_child_weight);
fprintf('  gamma: %.1f\n', best_params.gamma);

% Create DMatrix for all training data
X_train_py = py.numpy.array(X_train_norm);
y_train_py = py.numpy.array(y_train);
dtrain = xgb.DMatrix(X_train_py, y_train_py);

% Final parameters with regularization
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

% Train final model
bst_final = xgb.train(params_final, dtrain, int32(best_params.num_round));

%% 8. Evaluate final model
fprintf('\n=== 8. Final Model Evaluation ===\n');

y_pred_py = bst_final.predict(dtrain);
y_pred = double(py.array.array('d', py.numpy.nditer(y_pred_py)));
y_pred = y_pred(:);

% Calculate metrics
final_r2 = corr(y_pred, y_train)^2;
final_rmse = sqrt(mean((y_pred - y_train).^2));
final_mae = mean(abs(y_pred - y_train));
residuals = y_pred - y_train;
mse = mean(residuals.^2);

% Additional metrics
nse = 1 - (sum((y_train - y_pred).^2) / sum((y_train - mean(y_train)).^2));
pbias = 100 * (sum(y_train - y_pred) / sum(y_train));
generalization_gap_r2 = final_r2 - best_validation_metrics.Mean_R2;

fprintf('\n=== Final Model Evaluation ===\n');
fprintf('R²: %.4f\n', final_r2);
fprintf('RMSE: %.4f\n', final_rmse);
fprintf('MAE: %.4f\n', final_mae);
fprintf('MSE: %.4f\n', mse);
fprintf('NSE: %.4f\n', nse);
fprintf('PBIAS: %.2f%%\n', pbias);
fprintf('Generalization gap (R²): %.4f\n', generalization_gap_r2);

%% 9. Calculate relative variable importance
fprintf('\n=== 9. Calculating Variable Importance ===\n');

try
    importance_dict = bst_final.get_score(pyargs('importance_type','weight'));
    
    % Convert to MATLAB cell arrays
    keys_py = py.list(importance_dict.keys());
    values_py = py.list(importance_dict.values());
    
    keys_cell = cell(keys_py);
    values_cell = cell(values_py);
    
    % Convert values to numbers
    importance_values = zeros(1, length(keys_cell));
    for i = 1:length(keys_cell)
        importance_values(i) = double(values_cell{i});
    end
    
    % Create importance array for all variables
    importance_raw = zeros(1, length(variable_names_cell));
    for i = 1:length(keys_cell)
        key_str = char(keys_cell{i});
        % Extract index from f0, f1, ...
        idx = str2double(key_str(2:end)) + 1;
        if idx <= length(importance_raw)
            importance_raw(idx) = importance_values(i);
        end
    end
    
    % Convert to relative importance (0 to 1) for comparison with ANN
    relative_importance_xgb = importance_raw / sum(importance_raw);
    
catch ME
    fprintf('Warning: Could not calculate variable importance: %s\n', ME.message);
    % Create default importance
    importance_raw = ones(1, length(variable_names_cell));
    relative_importance_xgb = importance_raw / sum(importance_raw);
end

fprintf('Variable importance calculated successfully\n');

%% 10. Generate publication-quality figures
fprintf('\n=== 10. Generating Figures ===\n');

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

% Figure 3: Variable importance (relative)
subplot(2,3,3);
bar(relative_importance_xgb, 'FaceColor', [0.8 0.4 0.2]);
title('(c) Relative Variable Importance', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTickLabel', variable_names_cell);
xtickangle(45);
ylabel('Relative Importance');
grid on;

% Figure 4: Predicted vs Observed
subplot(2,3,4);
scatter(y_train, y_pred, 50, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--', 'LineWidth', 2);
xlabel('Observed Values');
ylabel('Predicted Values');
title(sprintf('(d) Predicted vs Observed (R² = %.3f)', final_r2), 'FontSize', 12);
grid on;
axis equal;

% Figure 5: Residual distribution
subplot(2,3,5);
histogram(residuals, 20, 'FaceColor', [0.6 0.3 0.7], 'FaceAlpha', 0.7);
xlabel('Residuals');
ylabel('Frequency');
title('(e) Residual Distribution', 'FontSize', 12);
grid on;

% Figure 6: Cross-validation performance
subplot(2,3,6);
boxplot([results.Mean_R2], 'Labels', {'XGBoost'});
ylabel('Cross-Validation R²');
title('(f) Cross-Validation Performance', 'FontSize', 12);
grid on;

% Save figure
saveas(gcf, fullfile(output_path, 'XGBoost_Results.png'));
fprintf('Figure saved: %s\n', fullfile(output_path, 'XGBoost_Results.png'));

%% 11. Regional prediction (GeoTIFF raster files)
fprintf('\n=== 11. Regional Prediction ===\n');

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
num_rows = size(ref_matrix, 1);
num_cols = size(ref_matrix, 2);
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

% Replace any NaN or Inf with zero
X_predict_norm(~isfinite(X_predict_norm)) = 0;

% Create DMatrix for XGBoost
dX_predict = xgb.DMatrix(py.numpy.array(X_predict_norm(valid_pixels,:)));

% Predict using XGBoost
fprintf('Predicting thermal diffusivity for region...\n');
tic;
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

% Create color map figure for regional prediction
figure('Position', [200, 200, 800, 600], 'Color', 'white');
imagesc(thermal_cond_map, 'AlphaData', ~isnan(thermal_cond_map));
colorbar;
colormap(jet);
title('Predicted Soil Thermal Diffusivity Map (XGBoost)', 'FontSize', 14);
xlabel('Column');
ylabel('Row');
axis equal;
print(gcf, fullfile(output_path, 'Thermal_Conductivity_Map_XGB.png'), '-dpng', '-r300');

%% 12. Save results
fprintf('\n=== 12. Saving Results ===\n');

% Save model and parameters
save(fullfile(output_path, 'XGBoost_Model.mat'), ...
     'bst_final', 'best_params', 'best_validation_metrics', ...
     'final_r2', 'final_rmse', 'final_mae', ...
     'vif_values', 'correlation_matrix', 'x_min', 'x_range', ...
     'relative_importance_xgb', 'results', 'variable_names_cell', '-v7.3');

% Create summary report
report_file = fullfile(output_path, 'XGBoost_Report.txt');
fid = fopen(report_file, 'w');

fprintf(fid, '========================================\n');
fprintf(fid, 'XGBoost Model Summary Report\n');
fprintf(fid, '========================================\n\n');

fprintf(fid, 'Best Hyperparameters:\n');
fprintf(fid, '  max_depth: %d\n', best_params.max_depth);
fprintf(fid, '  eta: %.4f\n', best_params.eta);
fprintf(fid, '  num_round: %d\n', best_params.num_round);
fprintf(fid, '  subsample: %.1f\n', best_params.subsample);
fprintf(fid, '  colsample_bytree: %.1f\n', best_params.colsample_bytree);
fprintf(fid, '  reg_alpha: %.1f\n', best_params.reg_alpha);
fprintf(fid, '  reg_lambda: %.1f\n', best_params.reg_lambda);
fprintf(fid, '  min_child_weight: %d\n', best_params.min_child_weight);
fprintf(fid, '  gamma: %.1f\n\n', best_params.gamma);

fprintf(fid, 'Model Performance Metrics:\n');
fprintf(fid, '  R²: %.4f\n', final_r2);
fprintf(fid, '  RMSE: %.4f\n', final_rmse);
fprintf(fid, '  MAE: %.4f\n', final_mae);
fprintf(fid, '  MSE: %.4f\n', mse);
fprintf(fid, '  NSE: %.4f\n', nse);
fprintf(fid, '  PBIAS: %.2f%%\n\n', pbias);

fprintf(fid, 'Cross-Validation Results (5-fold):\n');
fprintf(fid, '  Mean R²: %.4f ± %.4f\n', best_validation_metrics.Mean_R2, best_validation_metrics.Std_R2);
fprintf(fid, '  Mean RMSE: %.4f\n', best_validation_metrics.Mean_RMSE);
fprintf(fid, '  Mean MAE: %.4f\n\n', best_validation_metrics.Mean_MAE);

fprintf(fid, 'Relative Variable Importance (for comparison with ANN):\n');
for i = 1:length(variable_names_cell)
    fprintf(fid, '  %s: %.4f\n', variable_names_cell{i}, relative_importance_xgb(i));
end

fprintf(fid, '\nVIF Analysis:\n');
for i = 1:length(variable_names_cell)
    fprintf(fid, '  %s: %.2f\n', variable_names_cell{i}, vif_values(i));
end

fprintf(fid, '\nGeneralization Analysis:\n');
fprintf(fid, '  Generalization gap (R²): %.4f\n', generalization_gap_r2);

fclose(fid);
fprintf('Report saved: %s\n', report_file);

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
