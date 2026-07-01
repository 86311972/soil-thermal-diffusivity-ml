%% ========================================================================
% Random Forest (RF) for Soil Thermal Diffusivity Prediction
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
output_path = '';  % e.g., 'C:\YourResults\RandomForest\'

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
valid_samples = ~any(isnan(X_train_original), 2) & ~isnan(y_train);
X_train_original = X_train_original(valid_samples, :);
y_train = y_train(valid_samples);

fprintf('Number of training samples after cleaning: %d\n', length(y_train));

%% 2. Soil texture calculation and feature engineering
fprintf('\n=== 2. Feature Engineering ===\n');

clay = X_train_original(:,5);
sand = X_train_original(:,6);
silt = X_train_original(:,7);

% Calculate total and normalize to percentages
total = clay + sand + silt;
clay_percent = (clay ./ total) * 100;
sand_percent = (sand ./ total) * 100;
silt_percent = (silt ./ total) * 100;

% Calculate soil texture index (weighted average)
soil_texture_index = (clay_percent * 0.4 + sand_percent * 0.3 + silt_percent * 0.3);

% Create new feature matrix
X_train_new = X_train_original(:,1:4);
X_train_new = [X_train_new, X_train_original(:,8:9)];
X_train_new = [X_train_new, soil_texture_index];

% Variable names for plots
variable_names_cell = {'ndvi', 'sstad', 'sstay', 'bd', 'soc', 'wv', 'soil_texture'};
fprintf('Number of features after engineering: %d\n', size(X_train_new, 2));

%% 3. Min-Max Normalization
fprintf('\n=== 3. Data Normalization ===\n');

[X_train_norm, x_min, x_range] = normalize_minmax(X_train_new);
fprintf('Data normalized to [0, 1] range\n');

%% 4. Multicollinearity analysis and VIF calculation
fprintf('\n=== 4. Multicollinearity Analysis (VIF) ===\n');

correlation_matrix = corr(X_train_norm);

% Calculate VIF values
num_vars = size(X_train_norm, 2);
vif_values = zeros(1, num_vars);

for i = 1:num_vars
    other_vars = [1:i-1, i+1:num_vars];
    X_other = X_train_norm(:, other_vars);
    X_target = X_train_norm(:, i);
    
    beta = regress(X_target, [ones(size(X_other, 1), 1), X_other]);
    predicted = [ones(size(X_other, 1), 1), X_other] * beta;
    
    ss_res = sum((X_target - predicted).^2);
    ss_tot = sum((X_target - mean(X_target)).^2);
    r_squared = 1 - (ss_res / ss_tot);
    
    vif_values(i) = 1 / (1 - r_squared);
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

%% 5. Hyperparameter optimization with cross-validation
fprintf('\n=== 5. Hyperparameter Optimization (5-fold CV) ===\n');

% Optimized hyperparameters for 337 samples and 7 features
param_grid = struct();
param_grid.MinLeafSize = [25, 30, 35, 40];
param_grid.NumPredictorsToSample = [2, 3];
param_grid.NumTrees = [150, 200, 300];

% Storage for results
results = [];
combination_count = 0;
total_combinations = length(param_grid.MinLeafSize) * length(param_grid.NumPredictorsToSample) * length(param_grid.NumTrees);

fprintf('Total combinations to test: %d\n', total_combinations);

best_r2_cv = -Inf;
best_params = struct();
best_validation_metrics = struct();

for min_leaf = param_grid.MinLeafSize
    for num_pred = param_grid.NumPredictorsToSample
        for num_trees = param_grid.NumTrees
            combination_count = combination_count + 1;
            fprintf('Processing combination %d of %d (MinLeaf: %d, NumPred: %d, Trees: %d)\n', ...
                    combination_count, total_combinations, min_leaf, num_pred, num_trees);
            
            % 5-fold cross-validation with repeats
            k = 5;
            num_repeats = 5;
            all_cv_r2 = zeros(k * num_repeats, 1);
            all_cv_rmse = zeros(k * num_repeats, 1);
            all_cv_mae = zeros(k * num_repeats, 1);
            
            repeat_count = 0;
            for repeat = 1:num_repeats
                cv = cvpartition(size(X_train_norm,1), 'KFold', k);
                for fold = 1:k
                    repeat_count = repeat_count + 1;
                    idxTrain = training(cv, fold);
                    idxTest = test(cv, fold);
                    
                    model_cv = TreeBagger(num_trees, X_train_norm(idxTrain,:), y_train(idxTrain), ...
                                        'Method', 'regression', ...
                                        'MinLeafSize', min_leaf, ...
                                        'NumPredictorsToSample', num_pred, ...
                                        'OOBPrediction', 'off', ...
                                        'InBagFraction', 0.8);
                    
                    y_pred_cv = predict(model_cv, X_train_norm(idxTest,:));
                    all_cv_r2(repeat_count) = corr(y_pred_cv, y_train(idxTest))^2;
                    residuals = y_pred_cv - y_train(idxTest);
                    all_cv_rmse(repeat_count) = sqrt(mean(residuals.^2));
                    all_cv_mae(repeat_count) = mean(abs(residuals));
                end
            end
            
            % Average metrics
            mean_r2 = mean(all_cv_r2);
            mean_rmse = mean(all_cv_rmse);
            mean_mae = mean(all_cv_mae);
            std_r2 = std(all_cv_r2);
            std_rmse = std(all_cv_rmse);
            std_mae = std(all_cv_mae);
            
            % Store results
            result = struct();
            result.MinLeafSize = min_leaf;
            result.NumPredictorsToSample = num_pred;
            result.NumTrees = num_trees;
            result.Mean_R2 = mean_r2;
            result.Std_R2 = std_r2;
            result.Mean_RMSE = mean_rmse;
            result.Std_RMSE = std_rmse;
            result.Mean_MAE = mean_mae;
            result.Std_MAE = std_mae;
            
            results = [results; result];
            
            % Selection criteria: high R² and stability (std < 0.2)
            if mean_r2 > best_r2_cv && std_r2 < 0.2
                best_r2_cv = mean_r2;
                best_params.MinLeafSize = min_leaf;
                best_params.NumPredictorsToSample = num_pred;
                best_params.NumTrees = num_trees;
                best_validation_metrics.Mean_R2 = mean_r2;
                best_validation_metrics.Std_R2 = std_r2;
                best_validation_metrics.Mean_RMSE = mean_rmse;
                best_validation_metrics.Std_RMSE = std_rmse;
                best_validation_metrics.Mean_MAE = mean_mae;
                best_validation_metrics.Std_MAE = std_mae;
            end
        end
    end
end

% If no model met stability criteria, select best R²
if best_r2_cv == -Inf
    [~, best_idx] = max([results.Mean_R2]);
    best_r2_cv = results(best_idx).Mean_R2;
    best_params.MinLeafSize = results(best_idx).MinLeafSize;
    best_params.NumPredictorsToSample = results(best_idx).NumPredictorsToSample;
    best_params.NumTrees = results(best_idx).NumTrees;
    best_validation_metrics.Mean_R2 = results(best_idx).Mean_R2;
    best_validation_metrics.Std_R2 = results(best_idx).Std_R2;
    best_validation_metrics.Mean_RMSE = results(best_idx).Mean_RMSE;
    best_validation_metrics.Std_RMSE = results(best_idx).Std_RMSE;
    best_validation_metrics.Mean_MAE = results(best_idx).Mean_MAE;
    best_validation_metrics.Std_MAE = results(best_idx).Std_MAE;
end

%% 5.5 Display optimal hyperparameters
fprintf('\n========================================\n');
fprintf('   Optimal Random Forest Hyperparameters   \n');
fprintf('========================================\n');
fprintf('Number of Trees (NumTrees): %d\n', best_params.NumTrees);
fprintf('Minimum Leaf Size (MinLeafSize): %d\n', best_params.MinLeafSize);
fprintf('Predictors per Split (NumPredictorsToSample): %d\n', best_params.NumPredictorsToSample);
fprintf('Cross-validated R²: %.4f ± %.4f\n', ...
    best_validation_metrics.Mean_R2, best_validation_metrics.Std_R2);
fprintf('Cross-validated RMSE: %.4f\n', best_validation_metrics.Mean_RMSE);
fprintf('Cross-validated MAE: %.4f\n', best_validation_metrics.Mean_MAE);
fprintf('========================================\n');

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

%% 7. Train final model on training data
fprintf('\n=== 7. Training Final Random Forest Model ===\n');

final_model = TreeBagger(best_params.NumTrees, X_train_hold, y_train_hold, ...
                        'Method', 'regression', ...
                        'MinLeafSize', best_params.MinLeafSize, ...
                        'NumPredictorsToSample', best_params.NumPredictorsToSample, ...
                        'OOBPrediction', 'on', ...
                        'OOBPredictorImportance', 'on', ...
                        'InBagFraction', 0.8);

%% 8. Model evaluation on training data
fprintf('\n=== 8. Training Data Evaluation ===\n');

y_pred_train = predict(final_model, X_train_hold);
y_pred_train = double(y_pred_train);

% Training metrics
train_r2 = corr(y_pred_train, y_train_hold)^2;
train_rmse = sqrt(mean((y_pred_train - y_train_hold).^2));
train_mae = mean(abs(y_pred_train - y_train_hold));
train_mse = mean((y_pred_train - y_train_hold).^2);
train_nse = 1 - sum((y_train_hold - y_pred_train).^2) / sum((y_train_hold - mean(y_train_hold)).^2);
train_pbias = 100 * sum(y_train_hold - y_pred_train) / sum(y_train_hold);
train_residuals = y_pred_train - y_train_hold;

%% 9. Model evaluation on validation data
fprintf('\n=== 9. Validation Data Evaluation ===\n');

y_pred_val = predict(final_model, X_val_hold);
y_pred_val = double(y_pred_val);

% Validation metrics
val_r2 = corr(y_pred_val, y_val_hold)^2;
val_rmse = sqrt(mean((y_pred_val - y_val_hold).^2));
val_mae = mean(abs(y_pred_val - y_val_hold));
val_mse = mean((y_pred_val - y_val_hold).^2);
val_nse = 1 - sum((y_val_hold - y_pred_val).^2) / sum((y_val_hold - mean(y_val_hold)).^2);
val_pbias = 100 * sum(y_val_hold - y_pred_val) / sum(y_val_hold);
val_residuals = y_pred_val - y_val_hold;

%% 10. OOB error calculation
fprintf('\n=== 10. OOB Error Calculation ===\n');

oob_error_final = oobError(final_model, 'Mode', 'ensemble');
fprintf('OOB Error: %.4f\n', oob_error_final);

%% 11. Train model on full dataset for variable importance (for paper)
fprintf('\n=== 11. Training Full Model for Variable Importance ===\n');

final_model_full = TreeBagger(best_params.NumTrees, X_train_norm, y_train, ...
                            'Method', 'regression', ...
                            'MinLeafSize', best_params.MinLeafSize, ...
                            'NumPredictorsToSample', best_params.NumPredictorsToSample, ...
                            'OOBPrediction', 'on', ...
                            'OOBPredictorImportance', 'on', ...
                            'InBagFraction', 0.8);

% Variable importance on full dataset
raw_importance_full = final_model_full.OOBPermutedPredictorDeltaError;
relative_importance_rf_full = raw_importance_full / sum(raw_importance_full);

% Predict on full dataset for final R²
y_pred_full = predict(final_model_full, X_train_norm);
y_pred_full = double(y_pred_full);
final_r2_full = corr(y_pred_full, y_train)^2;

fprintf('R² on full dataset: %.4f\n', final_r2_full);

%% 12. Final performance summary table
fprintf('\n================================================================================\n');
fprintf('                  Random Forest Model Performance Summary                        \n');
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
fprintf('OOB Error |   %.4f    |      -      |       -\n', oob_error_final);
fprintf('================================================================================\n');

% Save results table as CSV
results_table = table(...
    [train_r2; train_rmse; train_mae; train_mse; train_nse; train_pbias; oob_error_final], ...
    [val_r2; val_rmse; val_mae; val_mse; val_nse; val_pbias; NaN], ...
    [best_validation_metrics.Mean_R2; best_validation_metrics.Mean_RMSE; best_validation_metrics.Mean_MAE; NaN; NaN; NaN; NaN], ...
    'VariableNames', {'Training', 'Validation', 'CV'}, ...
    'RowNames', {'R²', 'RMSE', 'MAE', 'MSE', 'NSE', 'PBIAS', 'OOB_Error'});
writetable(results_table, fullfile(output_path, 'RandomForest_Results.csv'), 'WriteRowNames', true);

%% 13. Display variable importance
fprintf('\n=== Relative Variable Importance (Full Dataset - 337 samples) ===\n');
fprintf('-------------------------------------------------------------\n');
for i = 1:length(variable_names_cell)
    fprintf('%s: %.4f\n', variable_names_cell{i}, relative_importance_rf_full(i));
end

% Save importance table
importance_table = table(variable_names_cell', relative_importance_rf_full', ...
    'VariableNames', {'Variable', 'Relative_Importance'});
writetable(importance_table, fullfile(output_path, 'RandomForest_Importance.csv'));

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

% Figure 3: Variable importance (full dataset)
subplot(2,3,3);
bar(relative_importance_rf_full, 'FaceColor', [0.8 0.4 0.2]);
title('(c) Variable Importance (Full Dataset)', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTickLabel', variable_names_cell);
xtickangle(45);
ylabel('Relative Importance');
grid on;

% Figure 4: Predicted vs Observed (Training)
subplot(2,3,4);
scatter(y_train_hold, y_pred_train, 50, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot([min(y_train_hold), max(y_train_hold)], [min(y_train_hold), max(y_train_hold)], 'r--', 'LineWidth', 2);
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

% Figure 6: OOB learning curve
subplot(2,3,6);
plot(oobError(final_model), 'LineWidth', 2, 'Color', [0.9 0.5 0.1]);
xlabel('Number of Trees');
ylabel('OOB Error');
title('(f) OOB Learning Curve', 'FontSize', 12);
grid on;

% Save figure
print(gcf, fullfile(output_path, 'RandomForest_Results.png'), '-dpng', '-r300');
fprintf('Figure saved: %s\n', fullfile(output_path, 'RandomForest_Results.png'));

%% 15. Save results
fprintf('\n=== 15. Saving Results ===\n');

save(fullfile(output_path, 'RandomForest_Model.mat'), ...
     'final_model', 'final_model_full', 'best_params', 'best_validation_metrics', ...
     'train_r2', 'val_r2', 'train_rmse', 'val_rmse', ...
     'train_mae', 'val_mae', 'oob_error_final', ...
     'vif_values', 'correlation_matrix', 'x_min', 'x_range', ...
     'results', 'raw_importance_full', 'relative_importance_rf_full', ...
     'train_nse', 'val_nse', 'train_pbias', 'val_pbias', '-v7.3');

% Save report
report_file = fullfile(output_path, 'RandomForest_Report.txt');
fid = fopen(report_file, 'w');

fprintf(fid, '========================================\n');
fprintf(fid, 'Random Forest Model Summary Report\n');
fprintf(fid, '========================================\n\n');

fprintf(fid, 'Optimal Hyperparameters:\n');
fprintf(fid, '  MinLeafSize: %d\n', best_params.MinLeafSize);
fprintf(fid, '  NumPredictorsToSample: %d\n', best_params.NumPredictorsToSample);
fprintf(fid, '  NumTrees: %d\n\n', best_params.NumTrees);

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
fprintf(fid, 'OOB Error |   %.4f    |      -      |       -\n', oob_error_final);
fprintf(fid, '--------------------------------------------------------------------------------\n\n');

fprintf(fid, 'Variable Importance:\n');
for i = 1:length(variable_names_cell)
    fprintf(fid, '  %s: %.4f\n', variable_names_cell{i}, relative_importance_rf_full(i));
end

fprintf(fid, '\nRaw Importance (OOBPermutedPredictorDeltaError):\n');
for i = 1:length(variable_names_cell)
    fprintf(fid, '  %s: %.4f\n', variable_names_cell{i}, raw_importance_full(i));
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
total_map(total_map == 0) = 1;

clay_percent_map = (clay_map ./ total_map) * 100;
sand_percent_map = (sand_map ./ total_map) * 100;
silt_percent_map = (silt_map ./ total_map) * 100;

soil_texture_map = (clay_percent_map * 0.4 + sand_percent_map * 0.3 + silt_percent_map * 0.3);

% Create feature matrix for region
X_predict_original = reshape(all_layers, [], num_vars);
X_predict_new = X_predict_original(:,1:4);
X_predict_new = [X_predict_new, X_predict_original(:,8:9)];
X_predict_new = [X_predict_new, reshape(soil_texture_map, [], 1)];

% Identify valid pixels
valid_pixels = ~any(isnan(X_predict_new), 2);
fprintf('Valid pixels for prediction: %d out of %d\n', sum(valid_pixels), length(valid_pixels));

% Normalize region data using training parameters
X_predict_norm = (X_predict_new(valid_pixels, :) - x_min) ./ x_range;

% Predict using Random Forest model
fprintf('Predicting thermal diffusivity for region...\n');
tic;
y_pred = zeros(size(X_predict_new, 1), 1, 'single');
y_pred(valid_pixels) = predict(final_model, X_predict_norm);
toc;

% Reconstruct full map
thermal_cond_map = reshape(y_pred, num_rows, num_cols);
thermal_cond_map(~valid_pixels) = NaN;

% Save predicted map
output_raster = fullfile(output_path, 'Thermal_Conductivity_RF.tif');
geotiffwrite(output_raster, thermal_cond_map, R, 'CoordRefSysCode', 4326);
fprintf('Regional prediction map saved: %s\n', output_raster);

% Create color map figure
figure('Position', [200, 200, 800, 600], 'Color', 'white');
imagesc(thermal_cond_map, 'AlphaData', ~isnan(thermal_cond_map));
colorbar;
colormap(jet);
title('Predicted Soil Thermal Diffusivity Map (Random Forest)', 'FontSize', 14);
xlabel('Column');
ylabel('Row');
axis equal;
print(gcf, fullfile(output_path, 'Thermal_Conductivity_Map_RF.png'), '-dpng', '-r300');

fprintf('\n========================================\n');
fprintf('Process completed successfully!\n');
fprintf('Results saved in: %s\n', output_path);
fprintf('========================================\n');

%% ========================================================================
% Helper Functions
% ========================================================================

function [data_norm, min_vals, range_vals] = normalize_minmax(data)
    % Min-Max normalization to [0, 1] range
    min_vals = min(data, [], 1);
    max_vals = max(data, [], 1);
    range_vals = max_vals - min_vals;
    range_vals(range_vals == 0) = 1;
    data_norm = (data - min_vals) ./ range_vals;
end
