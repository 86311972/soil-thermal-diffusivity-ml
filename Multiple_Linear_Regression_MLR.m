%% ========================================================================
% Multiple Linear Regression (MLR) Model for Soil Thermal Diffusivity Prediction
%
% This code accompanies the paper:
% "A Hybrid Machine Learning–Physics Approach for Retrieving Thermal Diffusivity, Simulating Soil Temperature, and Zoning Thermal Regimes in Iran"
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
output_path = '';  % e.g., 'C:\YourResults\MLR\'

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

% Extract variables
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
total(total == 0) = 1;
soil_texture_index = (clay ./ total) * 0.4 + (sand ./ total) * 0.3 + (silt ./ total) * 0.3;

% Create full feature matrix
X_train_full = X_train_original(:,1:4);
X_train_full = [X_train_full, X_train_original(:,8:9)];
X_train_full = [X_train_full, soil_texture_index];

% Variable names
variable_names_full = {'ndvi', 'sstad', 'sstay', 'bd', 'soc', 'wv', 'soil_texture'};
fprintf('Number of features: %d\n', size(X_train_full, 2));

%% 3. Correlation and statistical significance analysis
fprintf('\n=== 3. Correlation and Significance Analysis ===\n');

% Calculate correlation matrix and p-values
[corr_matrix, p_values] = corr([X_train_full, y_train]);

% Extract correlation with target
corr_with_target = corr_matrix(1:end-1, end);
p_values_target = p_values(1:end-1, end);

% Display correlation table
fprintf('\n=== Correlation with Target Variable ===\n');
fprintf('%-15s\t%-12s\t%-12s\t%-15s\n', 'Variable', 'Correlation', 'p-value', 'Significance');
fprintf('%-15s\t%-12s\t%-12s\t%-15s\n', '--------', '-----------', '-------', '------------');
for i = 1:length(variable_names_full)
    if p_values_target(i) <= 0.01
        sig = '*** (p<0.01)';
    elseif p_values_target(i) <= 0.05
        sig = '** (p<0.05)';
    elseif p_values_target(i) <= 0.1
        sig = '* (p<0.1)';
    else
        sig = 'ns';
    end
    fprintf('%-15s\t%-12.4f\t%-12.4f\t%-15s\n', ...
            variable_names_full{i}, corr_with_target(i), p_values_target(i), sig);
end

%% 4. Min-Max Normalization
fprintf('\n=== 4. Data Normalization ===\n');

[X_train_norm, x_min, x_range] = normalize_minmax(X_train_full);
[y_train_norm, y_min, y_range] = normalize_minmax(y_train);
fprintf('Data normalized to [0, 1] range\n');

%% 5. Feature selection using multivariate p-values
fprintf('\n=== 5. Feature Selection (Multivariate Analysis) ===\n');

% Train initial model with all variables
fprintf('Training initial model with all variables...\n');
initial_model = fitlm(X_train_norm, y_train_norm, 'VarNames', [variable_names_full, 'alfa_norm']);

% Extract p-values from multivariate model (not univariate!)
p_values_multivariate = initial_model.Coefficients.pValue(2:end); % Exclude intercept

% Select significant features based on multivariate model
significance_threshold = 0.1;
significant_features = find(p_values_multivariate < significance_threshold);

if isempty(significant_features)
    fprintf('Warning: No significant features found. Using all features...\n');
    significant_features = 1:length(variable_names_full);
else
    fprintf('Significant features in multivariate model:\n');
    for i = 1:length(significant_features)
        idx = significant_features(i);
        fprintf('  %s (p-value: %.4f)\n', variable_names_full{idx}, p_values_multivariate(idx));
    end
end

% Select features
X_train_selected = X_train_full(:, significant_features);
X_train_norm_selected = X_train_norm(:, significant_features);
variable_names_selected = variable_names_full(significant_features);

% Store normalization parameters for selected features
x_min_selected = x_min(significant_features);
x_range_selected = x_range(significant_features);

% Identify removed features
removed_features = setdiff(1:length(variable_names_full), significant_features);
variable_names_removed = variable_names_full(removed_features);
p_values_removed = p_values_multivariate(removed_features);

fprintf('Selected features: %d out of %d\n', length(variable_names_selected), length(variable_names_full));
if ~isempty(variable_names_removed)
    fprintf('Removed features: %s\n', strjoin(variable_names_removed, ', '));
end

%% 6. Train final linear regression model
fprintf('\n=== 6. Training Final Linear Regression Model ===\n');

linear_model = fitlm(X_train_norm_selected, y_train_norm, 'VarNames', [variable_names_selected, 'alfa_norm']);

% Extract coefficients
coefficients = linear_model.Coefficients.Estimate;
intercept_norm = coefficients(1);
slopes_norm = coefficients(2:end);

% Convert coefficients back to original scale
slopes_original = slopes_norm .* (y_range ./ x_range_selected');
intercept_original = y_min + y_range * intercept_norm - sum(slopes_original .* x_min_selected');

% Display final regression equation
fprintf('\n=== Final Regression Equation ===\n');
fprintf('y = %.6f', intercept_original);
for i = 1:length(slopes_original)
    if slopes_original(i) >= 0
        fprintf(' + %.6f * %s', slopes_original(i), variable_names_selected{i});
    else
        fprintf(' - %.6f * %s', abs(slopes_original(i)), variable_names_selected{i});
    end
end
fprintf('\n\n');

%% 7. Model evaluation on training data
fprintf('\n=== 7. Model Evaluation ===\n');

% Predict on training data
y_pred_norm = predict(linear_model, X_train_norm_selected);
y_pred_train = y_pred_norm * y_range + y_min;

% Calculate metrics
final_r2 = corr(y_pred_train, y_train)^2;
final_rmse = sqrt(mean((y_pred_train - y_train).^2));
final_mae = mean(abs(y_pred_train - y_train));
mse = mean((y_pred_train - y_train).^2);
residuals = y_pred_train - y_train;

% Additional metrics
nse = 1 - (sum((y_train - y_pred_train).^2) / sum((y_train - mean(y_train)).^2));
pbias = 100 * (sum(y_train - y_pred_train) / sum(y_train));

fprintf('R²: %.4f\n', final_r2);
fprintf('RMSE: %.4f\n', final_rmse);
fprintf('MAE: %.4f\n', final_mae);
fprintf('MSE: %.4f\n', mse);
fprintf('NSE: %.4f\n', nse);
fprintf('PBIAS: %.2f%%\n', pbias);

%% 8. Cross-validation
fprintf('\n=== 8. Cross-Validation (5-fold) ===\n');

k = 5;
num_repeats = 5;
all_cv_r2 = zeros(k * num_repeats, 1);
all_cv_rmse = zeros(k * num_repeats, 1);
all_cv_mae = zeros(k * num_repeats, 1);

repeat_count = 0;
for repeat = 1:num_repeats
    cv = cvpartition(length(y_train), 'KFold', k);
    for fold = 1:k
        repeat_count = repeat_count + 1;
        idxTrain = training(cv, fold);
        idxTest = test(cv, fold);
        
        X_train_fold = X_train_selected(idxTrain, :);
        y_train_fold = y_train(idxTrain);
        X_test_fold = X_train_selected(idxTest, :);
        
        % Normalize fold data
        [X_train_fold_norm, x_min_fold, x_range_fold] = normalize_minmax(X_train_fold);
        [y_train_fold_norm, y_min_fold, y_range_fold] = normalize_minmax(y_train_fold);
        
        % Train model on fold
        model_fold = fitlm(X_train_fold_norm, y_train_fold_norm);
        
        % Predict on test fold
        X_test_fold_norm = (X_test_fold - x_min_fold) ./ x_range_fold;
        y_pred_test_norm = predict(model_fold, X_test_fold_norm);
        y_pred_test = y_pred_test_norm * y_range_fold + y_min_fold;
        
        % Calculate metrics
        all_cv_r2(repeat_count) = corr(y_pred_test, y_train(idxTest))^2;
        residuals_cv = y_pred_test - y_train(idxTest);
        all_cv_rmse(repeat_count) = sqrt(mean(residuals_cv.^2));
        all_cv_mae(repeat_count) = mean(abs(residuals_cv));
    end
end

% Cross-validation metrics
cv_metrics.Mean_R2 = mean(all_cv_r2);
cv_metrics.Std_R2 = std(all_cv_r2);
cv_metrics.Mean_RMSE = mean(all_cv_rmse);
cv_metrics.Mean_MAE = mean(all_cv_mae);
generalization_gap = final_r2 - cv_metrics.Mean_R2;

fprintf('Cross-Validation Results:\n');
fprintf('  Mean R²: %.4f ± %.4f\n', cv_metrics.Mean_R2, cv_metrics.Std_R2);
fprintf('  Mean RMSE: %.4f\n', cv_metrics.Mean_RMSE);
fprintf('  Mean MAE: %.4f\n', cv_metrics.Mean_MAE);
fprintf('  Generalization gap (R²): %.4f\n', generalization_gap);

%% 9. Relative variable importance analysis
fprintf('\n=== 9. Variable Importance Analysis ===\n');

% Calculate standardized coefficients
X_std = std(X_train_norm_selected, 0, 1);
y_std = std(y_train_norm);
standardized_coeffs = abs(slopes_norm .* X_std' / y_std);
standardized_coeffs_rel = standardized_coeffs / sum(standardized_coeffs) * 100;

% Display variable importance
fprintf('\n=== Relative Variable Importance ===\n');
fprintf('%-15s\t%-20s\t%-15s\n', 'Variable', 'Relative Importance (%)', 'Std. Coefficient');
fprintf('%-15s\t%-20s\t%-15s\n', '--------', '---------------------', '---------------');
for i = 1:length(variable_names_selected)
    fprintf('%-15s\t%-20.2f\t%-15.4f\n', ...
            variable_names_selected{i}, ...
            standardized_coeffs_rel(i), ...
            standardized_coeffs(i));
end

%% 10. VIF analysis (multicollinearity)
fprintf('\n=== 10. Multicollinearity Analysis (VIF) ===\n');

vif_values = zeros(length(variable_names_selected), 1);
for i = 1:length(variable_names_selected)
    other_vars = [1:i-1, i+1:length(variable_names_selected)];
    X_other = X_train_norm_selected(:, other_vars);
    X_target = X_train_norm_selected(:, i);
    
    beta = regress(X_target, [ones(size(X_other,1),1), X_other]);
    predicted = [ones(size(X_other,1),1), X_other] * beta;
    r_squared = 1 - sum((X_target - predicted).^2) / sum((X_target - mean(X_target)).^2);
    vif_values(i) = 1 / (1 - r_squared);
    
    if vif_values(i) > 10
        status = 'Severe multicollinearity';
    elseif vif_values(i) > 5
        status = 'Moderate multicollinearity';
    else
        status = 'Acceptable';
    end
    fprintf('%s: VIF = %.2f (%s)\n', variable_names_selected{i}, vif_values(i), status);
end

%% 11. Generate publication-quality figures
fprintf('\n=== 11. Generating Figures ===\n');

% Calculate correlation matrix for selected features
corr_selected = corr(X_train_selected);

figure('Position', [100, 100, 1400, 1000], 'Color', 'white');

% Figure 1: Correlation matrix (selected features)
subplot(2,3,1);
imagesc(corr_selected);
colorbar;
colormap(jet);
set(gca, 'XTick', 1:length(variable_names_selected), ...
         'YTick', 1:length(variable_names_selected), ...
         'XTickLabel', variable_names_selected, ...
         'YTickLabel', variable_names_selected);
xtickangle(45);
title('(a) Correlation Matrix (Selected Features)', 'FontSize', 12, 'FontWeight', 'bold');

% Figure 2: Regression coefficients (all variables)
subplot(2,3,2);
% Create coefficient vector for all variables
all_coefficients = zeros(length(variable_names_full), 1);
all_coefficients(significant_features) = linear_model.Coefficients{2:end, 1};

bar(all_coefficients, 'FaceColor', [0.2 0.6 0.8]);
title('(b) Linear Regression Coefficients (All Variables)', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTick', 1:length(variable_names_full), 'XTickLabel', variable_names_full);
xtickangle(45);
ylabel('Coefficient Value');
grid on;

% Highlight removed features
hold on;
for i = 1:length(removed_features)
    bar(removed_features(i), all_coefficients(removed_features(i)), 'FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.5);
end
legend('Selected Features', 'Removed Features', 'Location', 'best');

% Figure 3: Variable importance
subplot(2,3,3);
[sorted_importance, sort_idx] = sort(standardized_coeffs_rel, 'descend');
sorted_names = variable_names_selected(sort_idx);
barh(sorted_importance, 'FaceColor', [0.8 0.4 0.2]);
set(gca, 'YTickLabel', sorted_names);
title('(c) Relative Variable Importance (%)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Relative Importance (%)');
grid on;

% Add value labels on bars
for i = 1:length(sorted_importance)
    text(sorted_importance(i) + 0.5, i, sprintf('%.1f%%', sorted_importance(i)), ...
         'FontSize', 9, 'VerticalAlignment', 'middle');
end

% Figure 4: Predicted vs Observed
subplot(2,3,4);
scatter(y_train, y_pred_train, 50, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--', 'LineWidth', 2);
xlabel('Observed Values');
ylabel('Predicted Values');
title(sprintf('(d) Predicted vs Observed (R² = %.3f)', final_r2), 'FontSize', 12);
grid on;
axis equal;

% Figure 5: Residuals analysis
subplot(2,3,5);
scatter(y_pred_train, residuals, 50, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot(xlim, [0, 0], 'r-', 'LineWidth', 2);
xlabel('Predicted Values');
ylabel('Residuals');
title('(e) Residuals vs Predicted Values', 'FontSize', 12);
grid on;

% Figure 6: Training vs Cross-validation performance
subplot(2,3,6);
metrics = [final_r2, final_rmse, final_mae; 
           cv_metrics.Mean_R2, cv_metrics.Mean_RMSE, cv_metrics.Mean_MAE]';
bar(metrics);
set(gca, 'XTickLabel', {'R²', 'RMSE', 'MAE'});
ylabel('Value');
title('(f) Training vs Cross-Validation Performance', 'FontSize', 12);
legend('Training', 'Cross-Validation', 'Location', 'best');
grid on;

% Save figure
saveas(gcf, fullfile(output_path, 'MLR_Results.png'));
fprintf('Figure saved: %s\n', fullfile(output_path, 'MLR_Results.png'));

%% 12. Regional prediction (GeoTIFF raster files)
fprintf('\n=== 12. Regional Prediction ===\n');

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
soil_texture_map = (clay_map ./ total_map) * 0.4 + (sand_map ./ total_map) * 0.3 + (silt_map ./ total_map) * 0.3;

% Create feature matrix for region
X_predict_original = reshape(all_layers, [], num_vars);
X_predict_new = X_predict_original(:,1:4);
X_predict_new = [X_predict_new, X_predict_original(:,8:9)];
X_predict_new = [X_predict_new, reshape(soil_texture_map, [], 1)];

% Select only significant features for regional prediction
X_predict_selected = X_predict_new(:, significant_features);

% Identify valid pixels
valid_pixels = ~any(isnan(X_predict_selected), 2);
fprintf('Valid pixels for prediction: %d out of %d\n', sum(valid_pixels), length(valid_pixels));

% Normalize region data using training parameters
X_predict_norm = (X_predict_selected(valid_pixels, :) - x_min_selected) ./ x_range_selected;

% Predict using Linear Regression
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

% Create color map figure for regional prediction
figure('Position', [200, 200, 800, 600], 'Color', 'white');
imagesc(thermal_cond_map, 'AlphaData', ~isnan(thermal_cond_map));
colorbar;
colormap(jet);
title('Predicted Soil Thermal Diffusivity Map (Multiple Linear Regression)', 'FontSize', 14);
xlabel('Column');
ylabel('Row');
axis equal;
print(gcf, fullfile(output_path, 'Thermal_Conductivity_Map_MLR.png'), '-dpng', '-r300');

%% 13. Save results
fprintf('\n=== 13. Saving Results ===\n');

% Save model and parameters
save(fullfile(output_path, 'MLR_Model.mat'), ...
     'linear_model', 'X_train_selected', 'y_train', ...
     'variable_names_selected', 'significant_features', ...
     'variable_names_removed', 'p_values_removed', ...
     'final_r2', 'final_rmse', 'final_mae', 'cv_metrics', ...
     'corr_matrix', 'p_values', 'vif_values', ...
     'intercept_original', 'slopes_original', ...
     'x_min_selected', 'x_range_selected', 'y_min', 'y_range', ...
     'standardized_coeffs_rel', 'variable_names_full', '-v7.3');

% Create summary report
report_file = fullfile(output_path, 'MLR_Report.txt');
fid = fopen(report_file, 'w');

fprintf(fid, '========================================\n');
fprintf(fid, 'Multiple Linear Regression (MLR) Report\n');
fprintf(fid, '========================================\n\n');

fprintf(fid, 'Final Regression Equation:\n');
fprintf(fid, 'y = %.6f', intercept_original);
for i = 1:length(slopes_original)
    if slopes_original(i) >= 0
        fprintf(fid, ' + %.6f * %s', slopes_original(i), variable_names_selected{i});
    else
        fprintf(fid, ' - %.6f * %s', abs(slopes_original(i)), variable_names_selected{i});
    end
end
fprintf(fid, '\n\n');

fprintf(fid, 'Model Specifications:\n');
fprintf(fid, '  Training samples: %d\n', length(y_train));
fprintf(fid, '  Selected features: %d\n', length(variable_names_selected));
fprintf(fid, '  Features used: %s\n', strjoin(variable_names_selected, ', '));
fprintf(fid, '  Features removed: %s\n', strjoin(variable_names_removed, ', '));
fprintf(fid, '  Degrees of freedom: %d\n', linear_model.DFE);
fprintf(fid, '  F-statistic: %.4f\n', linear_model.ModelFitVsNullModel.Fstat);
fprintf(fid, '  Model p-value: %.6f\n\n', linear_model.ModelFitVsNullModel.Pvalue);

fprintf(fid, 'Performance Metrics:\n');
fprintf(fid, '  R²: %.4f\n', final_r2);
fprintf(fid, '  RMSE: %.4f\n', final_rmse);
fprintf(fid, '  MAE: %.4f\n', final_mae);
fprintf(fid, '  MSE: %.4f\n', mse);
fprintf(fid, '  NSE: %.4f\n', nse);
fprintf(fid, '  PBIAS: %.2f%%\n\n', pbias);

fprintf(fid, 'Cross-Validation Results (5-fold, 5 repeats):\n');
fprintf(fid, '  Mean R²: %.4f ± %.4f\n', cv_metrics.Mean_R2, cv_metrics.Std_R2);
fprintf(fid, '  Mean RMSE: %.4f\n', cv_metrics.Mean_RMSE);
fprintf(fid, '  Mean MAE: %.4f\n\n', cv_metrics.Mean_MAE);

fprintf(fid, 'Regression Coefficients (Original Scale):\n');
fprintf(fid, '  Intercept: %.6f\n', intercept_original);
for i = 1:length(variable_names_selected)
    fprintf(fid, '  %s: %.6f\n', variable_names_selected{i}, slopes_original(i));
end

fprintf(fid, '\nRelative Variable Importance:\n');
for i = 1:length(variable_names_selected)
    fprintf(fid, '  %s: %.2f%%\n', variable_names_selected{i}, standardized_coeffs_rel(i));
end

fprintf(fid, '\nVIF Analysis:\n');
for i = 1:length(variable_names_selected)
    fprintf(fid, '  %s: %.2f\n', variable_names_selected{i}, vif_values(i));
end

fprintf(fid, '\nGeneralization Analysis:\n');
fprintf(fid, '  Generalization gap (R²): %.4f\n', generalization_gap);

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
    min_vals = min(data, [], 1);
    max_vals = max(data, [], 1);
    range_vals = max_vals - min_vals;
    range_vals(range_vals == 0) = 1;
    data_norm = (data - min_vals) ./ range_vals;
end
