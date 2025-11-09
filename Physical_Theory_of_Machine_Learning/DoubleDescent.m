% Shows double descent in test MSE vs model dimension
% using minimum-norm least squares with random features.

clear; close all; clc;
% rng(0);

%% Settings
p_train = 100;      % number of training samples
p_test  = 5000;     % number of test samples
D       = 400;      % ambient feature dimension (must >= max p)
sigma   = 0.5;      % noise level
n_rep   = 30;       % repetitions to smooth the curve

n_list = 5 : 5 : 400;     % model sizes (#parameters) to sweep
n_p    = length(n_list);

% test error
test_mse = zeros(n_p, 1);

%% Main loop
w_true  = 1 * randn(D,1);

for rep = 1 : n_rep
    % Generate random design and ground truth
    X_train = randn(p_train, D);
    X_test  = randn(p_test,  D);    
    
    y_train = X_train * w_true + sigma * randn(p_train, 1);
    y_test  = X_test  * w_true + sigma * randn(p_test, 1);
    
    for i = 1 : n_p
        n = n_list(i);
        
        % Use only first n features as the model
        Z_train = X_train(:, 1 : n);   % n x p_train
        Z_test  = X_test(:, 1 : n);    % n x p_test
        
        % Fit minimum-norm interpolating solution:
        % underparam (n <= p): standard OLS;
        % overparam (n > p): min-norm solution via dual form.
        if n <= p_train
            % (Z'Z)^{-1} Z' y
            w_hat = (Z_train' * Z_train) \ (Z_train' * y_train);
        else
            % min-norm: Z' (Z Z')^{-1} y
            ZZt = Z_train * Z_train';
            w_hat = Z_train' * (ZZt \ y_train);
        end
        
        y_pred = Z_test * w_hat;
        test_mse(i) = test_mse(i) + mean((y_pred - y_test) .^ 2);
    end
end

% Average over repetitions
test_mse = test_mse / n_rep;

%% Plot double descent
figure;
scatter(n_list / p_train, test_mse, '*', 'DisplayName', 'Empirical Generalization Error');
hold on;

% Mark interpolation threshold p = n_train
xline(1, '--', 'LineWidth', 1.5, 'DisplayName', '$n = p$');
ylim([0, 3000]);

xlabel('Parameter / Data $\gamma = n / p$', 'Interpreter', 'latex');
ylabel('Generalization Error');
legend('Interpreter', 'latex');
% set(gca, 'XScale', 'log');
grid on;
fontname('Times New Roman');
fontsize(20, 'points');
set(gcf, 'Color', [1, 1, 1]);

%% Theoretical double descent curve
% Theoretical Curve
theory_mse = zeros(D, 1);
for n = 1 : D
    gamma = n / p_train;
    if n >= p_train
        theory_mse(n) = (gamma - 1) / gamma * norm(w_true(1 : n)) ^ 2 + (sigma ^ 2 + norm(w_true(n + 1 : end)) ^ 2) / (gamma - 1) + norm(w_true(n + 1 : end)) ^ 2;
    else
        theory_mse(n) = gamma / (1 - gamma) * (sigma ^ 2 + norm(w_true(n + 1 : end)) ^ 2) + norm(w_true(n + 1 : end)) ^ 2;
    end
end

plot((1 : D) / p_train, theory_mse, 'LineWidth', 2, 'DisplayName', 'Theoretical Generalization Error');

% exportgraphics(gcf, 'doubleDescent2.pdf', 'ContentType', 'vector');