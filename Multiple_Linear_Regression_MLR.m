%% ========================================================================
% Multiple Linear Regression (MLR) for Soil Thermal Diffusivity Prediction
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

% Path to Excel file containing training data
training_data_file = '';  % e.g., 'C:\YourData\station_info_train.xlsx'

% Output directory for results
output_path = '';  % e.g., 'C:\YourResults\LinearRegression\'

% Create output directory if it doesn't exist
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

fprintf('Training data file: %s\n', training_data_file);
fprintf('Output directory: %s\n', output_path);

%% 1. Load and preprocess training data
fprintf('\n=== 1. Loading Training Data ===\n');

station_data = readtable(training_data_file);

% Extract variables (without LST and DEM)
X_original = table2array(station_data(:, {'ndvi', 'sstad', 'sstay', 'bd', ...
                     'clay', 'sand', 'silt', 'soc', 'wv'}));
y_original = station_data.alfa;

% Remove invalid values
valid_samples = ~any(isnan(X_original), 2) & ~isnan(y_original);
X_original = X_original(valid_samples, :);
y_original = y_original(valid_samples);

fprintf('Total number of samples: %d\n', length(y_original));

%% 2. Soil texture calculation and feature engineering
fprintf('\n=== 2. Feature Engineering ===\n');

clay = X_original(:,5);
sand = X_original(:,6);
silt = X_original(:,7);
total = clay + sand + silt;
total(total == 0) = 1;

% Calculate soil texture index (weighted average)
soil_texture_index = (clay ./ total) * 0.4 + (sand ./ total) * 0.3 + (silt ./ total) * 0.3;

% Create full feature matrix
X_full = [X_original(:,1:4), X_original(:,8:9), soil_texture_index];
variable_names_full = {'ndvi', 'sstad', 'sstay', 'bd', 'soc', 'wv', 'soil_texture'};
fprintf('Number of features after engineering: %d\n', size(X_full, 2));

%% 3. Data split for final evaluation (80-20)
fprintf('\n=== 3. Data Split (80%% Training, 20%% Validation) ===\n');

rng(42);
cv_holdout = cvpartition(length(y_original), 'HoldOut', 0.2);
idxTrain = training(cv_holdout);
idxVal = test(cv_holdout);

X_train_raw = X_full(idxTrain, :);
y_train_raw = y_original(idxTrain);
X_val_raw = X_full(idxVal, :);
y_val_raw = y_original(idxVal);

fprintf('Training samples: %d, Validation samples: %d\n', sum(idxTrain), sum(idxVal));

%% 4. Min-Max Normalization
fprintf('\n=== 4. Data Normalization ===\n');

[X_train_norm, x_min, x_range] = normalize_minmax(X_train_raw);
[y_train_norm, y_min, y_range] = normalize_minmax(y_train_raw);
X_val_norm = (X_val_raw - x_min) ./ x_range;
fprintf('Data normalized to [0, 1] range\n');

%% 5. Variable selection using multivariate p-value (threshold = 0.1)
fprintf('\n=== 5. Variable Selection (Multivariate p-value < 0.1) ===\n');

% Fit initial model with all predictors
initial_model = fitlm(X_train_norm, y_train_norm, ...
    'VarNames', [variable_names_full, 'alfa_norm']);

% Extract p-values from multivariate model
p_values = initial_model.Coefficients.pValue(2:end);
t_stats = initial_model.Coefficients.tStat(2:end);

% Select significant features (p < 0.1)
significant_threshold = 0.1;
significant_features = find(p_values < significant_threshold);

if isempty(significant_features)
    fprintf('Warning: No significant features found. Using all features.\n');
    significant_features = 1:length(variable_names_full);
end

X_train_selected = X_train_norm(:, significant_features);
X_val_selected = X_val_norm(:, significant_features);
variable_names_selected = variable_names_full(significant_features);

removed_features = setdiff(1:length(variable_names_full), significant_features);
variable_names_removed = variable_names_full(removed_features);

fprintf('Selected variables (%d):\n', length(variable_names_selected));
for i = 1:length(variable_names_selected)
    idx = significant_features(i);
    fprintf('  %d. %s (p = %.4f)\n', i, variable_names_selected{i}, p_values(idx));
end

if ~isempty(variable_names_removed)
    fprintf('\nRemoved variables (%d):\n', length(variable_names_removed));
    for i = 1:length(variable_names_removed)
        idx = removed_features(i);
        fprintf('  %d. %s (p = %.4f)\n', i, variable_names_removed{i}, p_values(idx));
    end
end

x_min_selected = x_min(significant_features);
x_range_selected = x_range(significant_features);

%% 6. Train final MLR model
fprintf('\n=== 6. Training Final MLR Model ===\n');

linear_model = fitlm(X_train_selected, y_train_norm, ...
    'VarNames', [variable_names_selected, 'alfa_norm']);

% Extract coefficients
coefficients = linear_model.Coefficients.Estimate;
intercept_norm = coefficients(1);
slopes_norm = coefficients(2:end);

% Convert coefficients to original scale
slopes_original = slopes_norm .* (y_range ./ x_range_selected');
intercept_original = y_min + y_range * intercept_norm - sum(slopes_original .* x_min_selected');

% Final p-values and t-statistics
p_values_final = linear_model.Coefficients.pValue(2:end);
t_stats_final = linear_model.Coefficients.tStat(2:end);

fprintf('\n=== Final Regression Equation ===\n');
fprintf('y = %.6f', intercept_original);
for i = 1:length(slopes_original)
    if slopes_original(i) >= 0
        fprintf(' + %.6f × %s', slopes_original(i), variable_names_selected{i});
    else
        fprintf(' - %.6f × %s', abs(slopes_original(i)), variable_names_selected{i});
    end
end
fprintf('\n\n');

%% 7. Model evaluation on training and validation data
fprintf('\n=== 7. Model Evaluation ===\n');

% Training predictions
y_pred_train_norm = predict(linear_model, X_train_selected);
y_pred_train = y_pred_train_norm * y_range + y_min;

% Training metrics
train_r2 = corr(y_pred_train, y_train_raw)^2;
train_rmse = sqrt(mean((y_pred_train - y_train_raw).^2));
train_mae = mean(abs(y_pred_train - y_train_raw));
train_mse = mean((y_pred_train - y_train_raw).^2);
train_nse = 1 - sum((y_train_raw - y_pred_train).^2) / sum((y_train_raw - mean(y_train_raw)).^2);
train_pbias = 100 * sum(y_train_raw - y_pred_train) / sum(y_train_raw);
train_residuals = y_pred_train - y_train_raw;

% Validation predictions
y_pred_val_norm = predict(linear_model, X_val_selected);
y_pred_val = y_pred_val_norm * y_range + y_min;

% Validation metrics
val_r2 = corr(y_pred_val, y_val_raw)^2;
val_rmse = sqrt(mean((y_pred_val - y_val_raw).^2));
val_mae = mean(abs(y_pred_val - y_val_raw));
val_mse = mean((y_pred_val - y_val_raw).^2);
val_nse = 1 - sum((y_val_raw - y_pred_val).^2) / sum((y_val_raw - mean(y_val_raw)).^2);
val_pbias = 100 * sum(y_val_raw - y_pred_val) / sum(y_val_raw);

%% 8. Cross-validation on full dataset (337 samples)
fprintf('\n=== 8. Cross-Validation (5-fold, 5 repeats) ===\n');

k = 5;
num_repeats = 5;
all_cv_r2 = zeros(k * num_repeats, 1);
all_cv_rmse = zeros(k * num_repeats, 1);
all_cv_mae = zeros(k * num_repeats, 1);

repeat_count = 0;
for repeat = 1:num_repeats
    cv = cvpartition(length(y_original), 'KFold', k);
    for fold = 1:k
        repeat_count = repeat_count + 1;
        idxTrain_fold = training(cv, fold);
        idxTest_fold = test(cv, fold);
        
        X_train_fold = X_full(idxTrain_fold, :);
        y_train_fold = y_original(idxTrain_fold);
        X_test_fold = X_full(idxTest_fold, :);
        y_test_fold = y_original(idxTest_fold);
        
        % Normalize fold data
        [X_train_fold_norm, x_min_fold, x_range_fold] = normalize_minmax(X_train_fold);
        [y_train_fold_norm, y_min_fold, y_range_fold] = normalize_minmax(y_train_fold);
        X_test_fold_norm = (X_test_fold - x_min_fold) ./ x_range_fold;
        
        % Select significant features
        X_train_fold_sel = X_train_fold_norm(:, significant_features);
        X_test_fold_sel = X_test_fold_norm(:, significant_features);
        
        % Train model
        model_fold = fitlm(X_train_fold_sel, y_train_fold_norm);
        
        % Predict
        y_pred_test_norm = predict(model_fold, X_test_fold_sel);
        y_pred_test = y_pred_test_norm * y_range_fold + y_min_fold;
        
        % Calculate metrics
        all_cv_r2(repeat_count) = corr(y_pred_test, y_test_fold)^2;
        residuals_cv = y_pred_test - y_test_fold;
        all_cv_rmse(repeat_count) = sqrt(mean(residuals_cv.^2));
        all_cv_mae(repeat_count) = mean(abs(residuals_cv));
    end
end

cv_metrics.Mean_R2 = mean(all_cv_r2);
cv_metrics.Std_R2 = std(all_cv_r2);
cv_metrics.Mean_RMSE = mean(all_cv_rmse);
cv_metrics.Std_RMSE = std(all_cv_rmse);
cv_metrics.Mean_MAE = mean(all_cv_mae);
cv_metrics.Std_MAE = std(all_cv_mae);

fprintf('CV R²: %.4f ± %.4f\n', cv_metrics.Mean_R2, cv_metrics.Std_R2);
fprintf('CV RMSE: %.4f ± %.4f\n', cv_metrics.Mean_RMSE, cv_metrics.Std_RMSE);
fprintf('CV MAE: %.4f ± %.4f\n', cv_metrics.Mean_MAE, cv_metrics.Std_MAE);

%% 9. Variable importance (standardized coefficients)
fprintf('\n=== 9. Variable Importance Analysis ===\n');

X_std = std(X_train_selected, 0, 1);
y_std = std(y_train_norm);

% Standardized coefficients with sign
standardized_coeffs_signed = slopes_norm .* X_std' / y_std;

% Relative importance (from absolute values)
standardized_coeffs_abs = abs(standardized_coeffs_signed);
standardized_coeffs_rel = standardized_coeffs_abs / sum(standardized_coeffs_abs);

fprintf('\n=== Standardized Coefficients (with sign) ===\n');
fprintf('-------------------------------------------------------------\n');
fprintf('%-20s | %-15s\n', 'Variable', 'Std. Coefficient');
fprintf('-------------------------------------------------------------\n');
for i = 1:length(variable_names_selected)
    fprintf('%-20s | %15.4f\n', variable_names_selected{i}, standardized_coeffs_signed(i));
end
fprintf('-------------------------------------------------------------\n');

fprintf('\n=== Relative Importance ===\n');
fprintf('-------------------------------------------------------------\n');
fprintf('%-20s | %-15s | %-12s\n', 'Variable', 'Importance', 'Rank');
fprintf('-------------------------------------------------------------\n');

% Sort by importance
[sorted_imp, sort_idx] = sort(standardized_coeffs_rel, 'descend');
for i = 1:length(sort_idx)
    fprintf('%-20s | %15.4f | %-12d\n', ...
        variable_names_selected{sort_idx(i)}, ...
        standardized_coeffs_rel(sort_idx(i)), i);
end
fprintf('-------------------------------------------------------------\n');
fprintf('Total relative importance: %.4f\n', sum(standardized_coeffs_rel));

%% 10. Final regression coefficients table
fprintf('\n=== 10. Final Regression Coefficients ===\n');
fprintf('------------------------------------------------------------------------------------\n');
fprintf('%-20s | %-12s | %-12s | %-12s | %-12s\n', ...
    'Variable', 'Coefficient', 'Rel. Imp.', 't-stat', 'p-value');
fprintf('------------------------------------------------------------------------------------\n');

% Intercept
fprintf('%-20s | %12.6f | %12s | %12s | %12s\n', ...
    '(Intercept)', intercept_original, '-', '-', '-');

for i = 1:length(variable_names_selected)
    sig = '';
    if p_values_final(i) < 0.001; sig = '***';
    elseif p_values_final(i) < 0.01; sig = '**';
    elseif p_values_final(i) < 0.05; sig = '*'; end
    fprintf('%-20s | %12.6f | %12.4f | %12.4f | %12.4f %s\n', ...
        variable_names_selected{i}, slopes_original(i), ...
        standardized_coeffs_rel(i), t_stats_final(i), p_values_final(i), sig);
end
fprintf('------------------------------------------------------------------------------------\n');

if ~isempty(variable_names_removed)
    fprintf('\nRemoved variables from final model:\n');
    for i = 1:length(variable_names_removed)
        idx = removed_features(i);
        fprintf('  %s (p = %.4f)\n', variable_names_removed{i}, p_values(idx));
    end
end

%% 11. Final performance summary table
fprintf('\n================================================================================\n');
fprintf('                  Linear Regression Model Performance Summary                    \n');
fprintf('================================================================================\n');
fprintf('Metric    |  Training   | Validation  |  CV (Mean ± Std)\n');
fprintf('--------------------------------------------------------------------------------\n');
fprintf('R²        |   %.4f    |   %.4f    |   %.4f ± %.4f\n', ...
    train_r2, val_r2, cv_metrics.Mean_R2, cv_metrics.Std_R2);
fprintf('RMSE      |   %.4f    |   %.4f    |   %.4f ± %.4f\n', ...
    train_rmse, val_rmse, cv_metrics.Mean_RMSE, cv_metrics.Std_RMSE);
fprintf('MAE       |   %.4f    |   %.4f    |   %.4f ± %.4f\n', ...
    train_mae, val_mae, cv_metrics.Mean_MAE, cv_metrics.Std_MAE);
fprintf('MSE       |   %.4f    |   %.4f    |       -\n', train_mse, val_mse);
fprintf('NSE       |   %.4f    |   %.4f    |       -\n', train_nse, val_nse);
fprintf('PBIAS     |   %.2f%%   |   %.2f%%   |       -\n', train_pbias, val_pbias);
fprintf('================================================================================\n');

%% 12. Save results
fprintf('\n=== 11. Saving Results ===\n');

% Results table
results_table = table(...
    [train_r2; train_rmse; train_mae; train_mse; train_nse; train_pbias], ...
    [val_r2; val_rmse; val_mae; val_mse; val_nse; val_pbias], ...
    [cv_metrics.Mean_R2; cv_metrics.Mean_RMSE; cv_metrics.Mean_MAE; NaN; NaN; NaN], ...
    'VariableNames', {'Training', 'Validation', 'CV'}, ...
    'RowNames', {'R²', 'RMSE', 'MAE', 'MSE', 'NSE', 'PBIAS'});
writetable(results_table, fullfile(output_path, 'LinearRegression_Results.csv'), 'WriteRowNames', true);

% Coefficients table
importance_table = table(...
    variable_names_selected', slopes_original, standardized_coeffs_rel, ...
    t_stats_final, p_values_final, ...
    'VariableNames', {'Variable', 'Coefficient', 'Relative_Importance', 't_statistic', 'p_value'});
writetable(importance_table, fullfile(output_path, 'LinearRegression_Coefficients.csv'));

% Save model
save(fullfile(output_path, 'LinearRegression_Model.mat'), ...
     'linear_model', 'variable_names_selected', 'variable_names_removed', ...
     'train_r2', 'val_r2', 'train_rmse', 'val_rmse', ...
     'cv_metrics', 'intercept_original', 'slopes_original', ...
     'standardized_coeffs_signed', 'standardized_coeffs_rel', ...
     'p_values_final', 'x_min_selected', 'x_range_selected', ...
     'y_min', 'y_range', '-v7.3');

% Save report
report_file = fullfile(output_path, 'LinearRegression_Report.txt');
fid = fopen(report_file, 'w');

fprintf(fid, '========================================\n');
fprintf(fid, 'Linear Regression Model Summary Report\n');
fprintf(fid, '========================================\n\n');

fprintf(fid, 'Final Regression Equation:\n');
fprintf(fid, 'y = %.6f', intercept_original);
for i = 1:length(slopes_original)
    if slopes_original(i) >= 0
        fprintf(fid, ' + %.6f × %s', slopes_original(i), variable_names_selected{i});
    else
        fprintf(fid, ' - %.6f × %s', abs(slopes_original(i)), variable_names_selected{i});
    end
end
fprintf(fid, '\n\n');

fprintf(fid, 'Performance Summary:\n');
fprintf(fid, '--------------------------------------------------------------------------------\n');
fprintf(fid, 'Metric    |  Training   | Validation  |  CV (Mean ± Std)\n');
fprintf(fid, '--------------------------------------------------------------------------------\n');
fprintf(fid, 'R²        |   %.4f    |   %.4f    |   %.4f ± %.4f\n', ...
    train_r2, val_r2, cv_metrics.Mean_R2, cv_metrics.Std_R2);
fprintf(fid, 'RMSE      |   %.4f    |   %.4f    |   %.4f ± %.4f\n', ...
    train_rmse, val_rmse, cv_metrics.Mean_RMSE, cv_metrics.Std_RMSE);
fprintf(fid, 'MAE       |   %.4f    |   %.4f    |   %.4f ± %.4f\n', ...
    train_mae, val_mae, cv_metrics.Mean_MAE, cv_metrics.Std_MAE);
fprintf(fid, 'MSE       |   %.4f    |   %.4f    |       -\n', train_mse, val_mse);
fprintf(fid, 'NSE       |   %.4f    |   %.4f    |       -\n', train_nse, val_nse);
fprintf(fid, 'PBIAS     |   %.2f%%   |   %.2f%%   |       -\n', train_pbias, val_pbias);
fprintf(fid, '--------------------------------------------------------------------------------\n\n');

fprintf(fid, 'Standardized Coefficients and Relative Importance:\n');
fprintf(fid, '------------------------------------------------------------------------------------\n');
fprintf(fid, '%-20s | %-15s | %-15s\n', 'Variable', 'Std. Coeff.', 'Rel. Importance');
fprintf(fid, '------------------------------------------------------------------------------------\n');
for i = 1:length(variable_names_selected)
    fprintf(fid, '%-20s | %15.4f | %15.4f\n', ...
        variable_names_selected{i}, standardized_coeffs_signed(i), standardized_coeffs_rel(i));
end
fprintf(fid, '------------------------------------------------------------------------------------\n');

fprintf(fid, '\nRemoved variables: %s\n', strjoin(variable_names_removed, ', '));

fclose(fid);

fprintf('\nAll results saved to: %s\n', output_path);
fprintf('\nProcess completed successfully!\n');

%% 13. Generate publication-quality figures
fprintf('\n=== 12. Generating Figures ===\n');

figure('Position', [100, 100, 1400, 1000], 'Color', 'white');

% Figure 1: Relative importance
subplot(2,3,1);
bar(standardized_coeffs_rel, 'FaceColor', [0.8 0.4 0.2]);
title('(a) Variable Relative Importance', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTickLabel', variable_names_selected);
xtickangle(45);
ylabel('Relative Importance');
grid on;

% Figure 2: Predicted vs Observed (Training)
subplot(2,3,2);
scatter(y_train_raw, y_pred_train, 50, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot([min(y_train_raw), max(y_train_raw)], [min(y_train_raw), max(y_train_raw)], 'r--', 'LineWidth', 2);
xlabel('Observed Values');
ylabel('Predicted Values');
title(sprintf('(b) Training: R² = %.3f', train_r2), 'FontSize', 12);
grid on;

% Figure 3: Residuals vs Predicted
subplot(2,3,3);
scatter(y_pred_train, train_residuals, 50, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot(xlim, [0, 0], 'r-', 'LineWidth', 2);
xlabel('Predicted Values');
ylabel('Residuals');
title('(c) Residuals Analysis', 'FontSize', 12);
grid on;

% Figure 4: Performance comparison
subplot(2,3,4);
metrics = [train_r2, val_r2; train_rmse, val_rmse; train_mae, val_mae]';
bar(metrics);
set(gca, 'XTickLabel', {'R²', 'RMSE', 'MAE'});
ylabel('Value');
title('(d) Training vs Validation Performance', 'FontSize', 12);
legend('Training', 'Validation', 'Location', 'best');
grid on;

% Figure 5: Residual distribution
subplot(2,3,5);
histogram(train_residuals, 20, 'FaceColor', [0.6 0.3 0.7], 'FaceAlpha', 0.7);
xlabel('Residuals');
ylabel('Frequency');
title('(e) Residual Distribution', 'FontSize', 12);
grid on;

% Figure 6: Coefficients
subplot(2,3,6);
bar(slopes_original, 'FaceColor', [0.2 0.6 0.8]);
title('(f) Final Regression Coefficients', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTickLabel', variable_names_selected);
xtickangle(45);
ylabel('Coefficient Value');
grid on;

% Save figure
print(gcf, fullfile(output_path, 'LinearRegression_Figures.png'), '-dpng', '-r300');
fprintf('Figure saved: %s\n', fullfile(output_path, 'LinearRegression_Figures.png'));

%% 14. Regional prediction (GeoTIFF raster files)
fprintf('\n=== 13. Regional Prediction ===\n');

% List of raster files (USER MUST MATCH THESE NAMES TO THEIR FILES)
raster_names = {'ndvi_yearly', 'ssta_day', 'ssta_yearly', ...
                'bd_mean_50cm', 'clay_mean_50cm', 'sand_mean_50cm', ...
                'silt_mean_50cm', 'soc_mean_50cm', 'wv_mean_50cm'};

% Path to folder containing raster files
input_data_path = '';  % e.g., 'C:\YourData\rasters\'

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

soil_texture_map = (clay_map ./ total_map) * 0.4 + (sand_map ./ total_map) * 0.3 + (silt_map ./ total_map) * 0.3;

% Create feature matrix for region
X_predict_original = reshape(all_layers, [], num_vars);
X_predict_new = X_predict_original(:,1:4);
X_predict_new = [X_predict_new, X_predict_original(:,8:9)];
X_predict_new = [X_predict_new, reshape(soil_texture_map, [], 1)];

% Select only significant features
X_predict_selected = X_predict_new(:, significant_features);

% Identify valid pixels
valid_pixels = ~any(isnan(X_predict_selected), 2);
fprintf('Valid pixels for prediction: %d out of %d\n', sum(valid_pixels), length(valid_pixels));

% Normalize region data using training parameters
X_predict_norm = (X_predict_selected(valid_pixels, :) - x_min_selected) ./ x_range_selected;

% Predict using linear regression model
fprintf('Predicting thermal diffusivity for region...\n');
tic;
y_pred_region_norm = predict(linear_model, X_predict_norm);
y_pred_region = y_pred_region_norm * y_range + y_min;
toc;

% Reconstruct full map
y_pred = zeros(size(X_predict_selected, 1), 1, 'single');
y_pred(valid_pixels) = y_pred_region;
thermal_cond_map = reshape(y_pred, num_rows, num_cols);
thermal_cond_map(~valid_pixels) = NaN;

% Save predicted map
output_raster = fullfile(output_path, 'Thermal_Conductivity_MLR.tif');
geotiffwrite(output_raster, thermal_cond_map, R, 'CoordRefSysCode', 4326);
fprintf('Regional prediction map saved: %s\n', output_raster);

% Create color map figure
figure('Position', [200, 200, 800, 600], 'Color', 'white');
imagesc(thermal_cond_map, 'AlphaData', ~isnan(thermal_cond_map));
colorbar;
colormap(jet);
title('Predicted Soil Thermal Diffusivity Map (MLR)', 'FontSize', 14);
xlabel('Column');
ylabel('Row');
axis equal;
print(gcf, fullfile(output_path, 'Thermal_Conductivity_Map_MLR.png'), '-dpng', '-r300');

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
    zero_range = range_vals == 0;
    range_vals(zero_range) = 1;
    data_norm = (data - min_vals) ./ range_vals;
    data_norm(:, zero_range) = 0;
end
