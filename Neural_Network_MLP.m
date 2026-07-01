%% ========================================================================
% Neural Network (MLP) for Soil Thermal Diffusivity Prediction
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
output_path = '';  % e.g., 'C:\YourResults\ANN\'

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
[y_train_norm, y_min, y_range] = normalize_minmax(y_train);
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

% Parameter grid
param_grid = struct();
param_grid.HiddenLayerSizes = {3, 5, 7, [5,3]};
param_grid.ActivationFunctions = {'poslin', 'tansig'};
param_grid.LearningRates = [0.005, 0.001];
param_grid.MaxEpochs = [600, 800];
param_grid.Regularization = [0.01, 0.05];

% Storage for results
results = [];
combination_count = 0;
total_combinations = length(param_grid.HiddenLayerSizes) * length(param_grid.ActivationFunctions) * ...
                     length(param_grid.LearningRates) * length(param_grid.MaxEpochs) * ...
                     length(param_grid.Regularization);

fprintf('Total combinations to test: %d\n', total_combinations);

best_r2_cv = -Inf;
best_params = struct();
best_validation_metrics = struct();

for hidden_idx = 1:length(param_grid.HiddenLayerSizes)
    hidden_size = param_grid.HiddenLayerSizes{hidden_idx};
    for act_idx = 1:length(param_grid.ActivationFunctions)
        activation_func = param_grid.ActivationFunctions{act_idx};
        for lr_idx = 1:length(param_grid.LearningRates)
            lr = param_grid.LearningRates(lr_idx);
            for epoch_idx = 1:length(param_grid.MaxEpochs)
                max_epochs = param_grid.MaxEpochs(epoch_idx);
                for reg_idx = 1:length(param_grid.Regularization)
                    regularization = param_grid.Regularization(reg_idx);
                    
                    combination_count = combination_count + 1;
                    fprintf('Processing combination %d of %d\n', combination_count, total_combinations);
                    
                    % 5-fold cross-validation
                    k = 5;
                    cv = cvpartition(size(X_train_norm,1), 'KFold', k);
                    cv_r2 = zeros(k, 1);
                    cv_rmse = zeros(k, 1);
                    cv_mae = zeros(k, 1);
                    
                    for fold = 1:k
                        try
                            idxTrain = training(cv, fold);
                            idxTest = test(cv, fold);
                            
                            % Create and configure neural network
                            net = fitnet(hidden_size);
                            net.trainFcn = 'trainlm';
                            
                            % Set training parameters
                            net.trainParam.lr = lr;
                            net.trainParam.epochs = max_epochs;
                            net.trainParam.showWindow = false;
                            net.trainParam.showCommandLine = false;
                            
                            % Set regularization
                            net.performFcn = 'msereg';
                            net.performParam.regularization = regularization;
                            
                            % Set activation function for hidden layers
                            for layer_idx = 1:length(hidden_size)
                                net.layers{layer_idx}.transferFcn = activation_func;
                            end
                            
                            % Configure data splitting
                            net.divideFcn = 'divideind';
                            net.divideParam.trainInd = find(idxTrain);
                            net.divideParam.valInd = find(idxTest);
                            net.divideParam.testInd = [];
                            
                            % Train network
                            [net, ~] = train(net, X_train_norm', y_train_norm');
                            
                            % Predict on validation fold
                            y_pred_norm = net(X_train_norm(idxTest,:)');
                            y_pred = denormalize_minmax(y_pred_norm', y_min, y_range);
                            
                            % Calculate metrics
                            cv_r2(fold) = corr(y_pred, y_train(idxTest))^2;
                            residuals = y_pred - y_train(idxTest);
                            cv_rmse(fold) = sqrt(mean(residuals.^2));
                            cv_mae(fold) = mean(abs(residuals));
                            
                        catch ME
                            fprintf('Error in fold %d: %s\n', fold, ME.message);
                            cv_r2(fold) = 0;
                            cv_rmse(fold) = inf;
                            cv_mae(fold) = inf;
                        end
                    end
                    
                    % Average metrics (excluding invalid folds)
                    valid_folds = cv_r2 > 0;
                    if sum(valid_folds) > 0
                        mean_r2 = mean(cv_r2(valid_folds));
                        mean_rmse = mean(cv_rmse(valid_folds));
                        mean_mae = mean(cv_mae(valid_folds));
                        std_r2 = std(cv_r2(valid_folds));
                        std_rmse = std(cv_rmse(valid_folds));
                        std_mae = std(cv_mae(valid_folds));
                    else
                        mean_r2 = 0;
                        mean_rmse = inf;
                        mean_mae = inf;
                        std_r2 = 0;
                        std_rmse = 0;
                        std_mae = 0;
                    end
                    
                    % Store results
                    result = struct();
                    result.HiddenLayerSize = hidden_size;
                    result.ActivationFunction = activation_func;
                    result.LearningRate = lr;
                    result.MaxEpochs = max_epochs;
                    result.Regularization = regularization;
                    result.Mean_R2 = mean_r2;
                    result.Std_R2 = std_r2;
                    result.Mean_RMSE = mean_rmse;
                    result.Std_RMSE = std_rmse;
                    result.Mean_MAE = mean_mae;
                    result.Std_MAE = std_mae;
                    
                    results = [results; result];
                    
                    % Update best model
                    if mean_r2 > best_r2_cv && mean_r2 > 0
                        best_r2_cv = mean_r2;
                        best_params.HiddenLayerSize = hidden_size;
                        best_params.ActivationFunction = activation_func;
                        best_params.LearningRate = lr;
                        best_params.MaxEpochs = max_epochs;
                        best_params.Regularization = regularization;
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
    end
end

%% 5.5 Display optimal hyperparameters
fprintf('\n========================================\n');
fprintf('   Optimal Neural Network Hyperparameters   \n');
fprintf('========================================\n');
fprintf('Hidden layer structure: [%s]\n', num2str(best_params.HiddenLayerSize));
fprintf('Activation function: %s\n', best_params.ActivationFunction);
fprintf('Learning rate: %.4f\n', best_params.LearningRate);
fprintf('Maximum epochs: %d\n', best_params.MaxEpochs);
fprintf('Regularization: %.3f\n', best_params.Regularization);
fprintf('Cross-validated R²: %.4f ± %.4f\n', ...
    best_validation_metrics.Mean_R2, best_validation_metrics.Std_R2);
fprintf('========================================\n');

%% 6. Train final model with best hyperparameters
fprintf('\n=== 6. Training Final Model ===\n');

fprintf('Best hyperparameters:\n');
fprintf('  Hidden layer size: [%s]\n', num2str(best_params.HiddenLayerSize));
fprintf('  Activation function: %s\n', best_params.ActivationFunction);
fprintf('  Learning rate: %.4f\n', best_params.LearningRate);
fprintf('  Max epochs: %d\n', best_params.MaxEpochs);
fprintf('  Regularization: %.3f\n', best_params.Regularization);

% Create final network
final_net = fitnet(best_params.HiddenLayerSize);
final_net.trainFcn = 'trainlm';

% Set regularization for final model
final_net.performFcn = 'msereg';
final_net.performParam.regularization = best_params.Regularization;

% Set activation function for hidden layers
for layer_idx = 1:length(best_params.HiddenLayerSize)
    final_net.layers{layer_idx}.transferFcn = best_params.ActivationFunction;
end

% Set training parameters
final_net.trainParam.lr = best_params.LearningRate;
final_net.trainParam.epochs = best_params.MaxEpochs;
final_net.trainParam.showWindow = true;

% Configure data split for final model
final_net.divideFcn = 'dividerand';
final_net.divideParam.trainRatio = 0.8;
final_net.divideParam.valRatio = 0.2;
final_net.divideParam.testRatio = 0;

% Train final model
[final_net, tr_final] = train(final_net, X_train_norm', y_train_norm');

% Predict on training data
y_pred_norm_final = final_net(X_train_norm');
y_pred_final = denormalize_minmax(y_pred_norm_final', y_min, y_range);

% Final evaluation metrics
final_r2 = corr(y_pred_final, y_train)^2;
final_rmse = sqrt(mean((y_pred_final - y_train).^2));
final_mae = mean(abs(y_pred_final - y_train));
residuals = y_pred_final - y_train;
mse = mean(residuals.^2);

% Additional metrics
nse = 1 - (sum((y_train - y_pred_final).^2) / sum((y_train - mean(y_train)).^2));
pbias = 100 * (sum(y_train - y_pred_final) / sum(y_train));
generalization_gap_r2 = final_r2 - best_validation_metrics.Mean_R2;

fprintf('\n=== Final Model Evaluation ===\n');
fprintf('R²: %.4f\n', final_r2);
fprintf('RMSE: %.4f\n', final_rmse);
fprintf('MAE: %.4f\n', final_mae);
fprintf('MSE: %.4f\n', mse);
fprintf('NSE: %.4f\n', nse);
fprintf('PBIAS: %.2f%%\n', pbias);
fprintf('Generalization gap (R²): %.4f\n', generalization_gap_r2);

%% 7. Extract training and validation indices from final model
fprintf('\n=== 7. Training/Validation Split ===\n');

train_idx = tr_final.trainInd;
val_idx = tr_final.valInd;

fprintf('Training samples: %d, Validation samples: %d\n', length(train_idx), length(val_idx));

% Predict on training data
X_train_selected = X_train_norm(train_idx, :)';
y_pred_train_norm = final_net(X_train_selected);
y_pred_train = denormalize_minmax(y_pred_train_norm', y_min, y_range);

% Predict on validation data
X_val_selected = X_train_norm(val_idx, :)';
y_pred_val_norm = final_net(X_val_selected);
y_pred_val = denormalize_minmax(y_pred_val_norm', y_min, y_range);

% True values
y_true_train = y_train(train_idx);
y_true_val = y_train(val_idx);

% Training metrics
train_r2 = corr(y_pred_train, y_true_train)^2;
train_rmse = sqrt(mean((y_pred_train - y_true_train).^2));
train_mae = mean(abs(y_pred_train - y_true_train));
train_mse = mean((y_pred_train - y_true_train).^2);
train_nse = 1 - sum((y_true_train - y_pred_train).^2) / sum((y_true_train - mean(y_true_train)).^2);
train_pbias = 100 * sum(y_true_train - y_pred_train) / sum(y_true_train);

% Validation metrics
val_r2 = corr(y_pred_val, y_true_val)^2;
val_rmse = sqrt(mean((y_pred_val - y_true_val).^2));
val_mae = mean(abs(y_pred_val - y_true_val));
val_mse = mean((y_pred_val - y_true_val).^2);
val_nse = 1 - sum((y_true_val - y_pred_val).^2) / sum((y_true_val - mean(y_true_val)).^2);
val_pbias = 100 * sum(y_true_val - y_pred_val) / sum(y_true_val);

%% 8. Final performance summary table
fprintf('\n================================================================================\n');
fprintf('                  Neural Network Model Performance Summary                      \n');
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
writetable(results_table, fullfile(output_path, 'NeuralNetwork_Results.csv'), 'WriteRowNames', true);

%% 9. Generate publication-quality figures
fprintf('\n=== 9. Generating Figures ===\n');

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

% Figure 3: Learning curve
subplot(2,3,3);
plot(tr_final.perf, 'b-', 'LineWidth', 2, 'DisplayName', 'Training');
hold on;
plot(tr_final.vperf, 'r-', 'LineWidth', 2, 'DisplayName', 'Validation');
xlabel('Epochs');
ylabel('Mean Squared Error');
title('(c) Neural Network Learning Curve', 'FontSize', 12, 'FontWeight', 'bold');
legend('show');
grid on;

% Figure 4: Predicted vs Observed
subplot(2,3,4);
scatter(y_train, y_pred_final, 50, 'filled', 'MarkerFaceAlpha', 0.6);
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

% Figure 6: Variable importance
subplot(2,3,6);
importance_ann = compute_ann_importance(final_net, X_train_norm, y_train_norm, y_min, y_range);
bar(importance_ann, 'FaceColor', [0.8 0.4 0.2]);
title('(f) Variable Importance (Sensitivity Analysis)', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTickLabel', variable_names_cell);
xtickangle(45);
ylabel('Relative Importance');
grid on;

% Save figure
print(gcf, fullfile(output_path, 'NeuralNetwork_Results.png'), '-dpng', '-r300');
fprintf('Figure saved: %s\n', fullfile(output_path, 'NeuralNetwork_Results.png'));

%% 10. Save results
fprintf('\n=== 10. Saving Results ===\n');

save(fullfile(output_path, 'NeuralNetwork_Model.mat'), ...
     'final_net', 'best_params', 'best_validation_metrics', ...
     'final_r2', 'final_rmse', 'final_mae', 'tr_final', ...
     'train_r2', 'val_r2', 'train_rmse', 'val_rmse', ...
     'train_mae', 'val_mae', 'vif_values', 'correlation_matrix', ...
     'x_min', 'x_range', 'y_min', 'y_range', 'results', '-v7.3');

% Save report
report_file = fullfile(output_path, 'NeuralNetwork_Report.txt');
fid = fopen(report_file, 'w');

fprintf(fid, '========================================\n');
fprintf(fid, 'Neural Network Model Summary Report\n');
fprintf(fid, '========================================\n\n');

fprintf(fid, 'Best Hyperparameters:\n');
fprintf(fid, '  Hidden layer structure: [%s]\n', num2str(best_params.HiddenLayerSize));
fprintf(fid, '  Activation function: %s\n', best_params.ActivationFunction);
fprintf(fid, '  Training function: %s\n', 'trainlm');
fprintf(fid, '  Learning rate: %.4f\n', best_params.LearningRate);
fprintf(fid, '  Maximum epochs: %d\n', best_params.MaxEpochs);
fprintf(fid, '  Regularization parameter: %.3f\n\n', best_params.Regularization);

% Calculate number of parameters
input_size = 7;
output_size = 1;
if isscalar(best_params.HiddenLayerSize)
    num_params = (input_size * best_params.HiddenLayerSize) + (best_params.HiddenLayerSize * output_size) + best_params.HiddenLayerSize + output_size;
else
    num_params = (input_size * best_params.HiddenLayerSize(1)) + (best_params.HiddenLayerSize(1) * best_params.HiddenLayerSize(2)) + ...
                (best_params.HiddenLayerSize(2) * output_size) + sum(best_params.HiddenLayerSize) + output_size;
end
fprintf(fid, 'Number of parameters: ~%d\n', num_params);
fprintf(fid, 'Data-to-parameter ratio: %.2f:1\n\n', length(y_train)/num_params);

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

fprintf(fid, 'Variable Importance (Sensitivity Analysis):\n');
for i = 1:length(variable_names_cell)
    fprintf(fid, '  %s: %.4f\n', variable_names_cell{i}, importance_ann(i));
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

fprintf(fid, '\nGeneralization Analysis:\n');
fprintf(fid, '  Generalization gap (R²): %.4f\n', generalization_gap_r2);

fclose(fid);

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
}

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

% Predict using neural network
fprintf('Predicting thermal diffusivity for region...\n');
tic;
y_pred_norm_region = final_net(X_predict_norm');
y_pred_region = denormalize_minmax(y_pred_norm_region', y_min, y_range);
toc;

% Reconstruct full map
y_pred = zeros(size(X_predict_new, 1), 1, 'single');
y_pred(valid_pixels) = y_pred_region;
thermal_cond_map = reshape(y_pred, num_rows, num_cols);
thermal_cond_map(~valid_pixels) = NaN;

% Save predicted map
output_raster = fullfile(output_path, 'Thermal_Conductivity_ANN.tif');
geotiffwrite(output_raster, thermal_cond_map, R, 'CoordRefSysCode', 4326);
fprintf('Regional prediction map saved: %s\n', output_raster);

% Create color map figure
figure('Position', [200, 200, 800, 600], 'Color', 'white');
imagesc(thermal_cond_map, 'AlphaData', ~isnan(thermal_cond_map));
colorbar;
colormap(jet);
title('Predicted Soil Thermal Diffusivity Map (ANN)', 'FontSize', 14);
xlabel('Column');
ylabel('Row');
axis equal;
print(gcf, fullfile(output_path, 'Thermal_Conductivity_Map_ANN.png'), '-dpng', '-r300');

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

function data_denorm = denormalize_minmax(data_norm, min_vals, range_vals)
    % Denormalize data from [0, 1] back to original range
    data_denorm = data_norm .* range_vals + min_vals;
end

function importance = compute_ann_importance(net, X, ~, y_min, y_range)
    % Compute variable importance using sensitivity analysis
    num_vars = size(X, 2);
    base_pred_norm = net(X');
    base_pred = base_pred_norm' .* y_range + y_min;
    
    importance = zeros(1, num_vars);
    
    for i = 1:num_vars
        X_perturbed = X;
        X_perturbed(:, i) = X_perturbed(:, i) + 0.1 * std(X(:, i));
        
        pred_perturbed_norm = net(X_perturbed');
        pred_perturbed = pred_perturbed_norm' .* y_range + y_min;
        
        changes = abs(pred_perturbed - base_pred);
        importance(i) = mean(changes);
    end
    
    importance = importance / sum(importance);
end
