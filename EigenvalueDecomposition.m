clear; close all; clc;

%% Initialization
m = 7;
A = rand(m,m); A = A' + A;
B = diag(1:m);
X0 = eye(m);

stepSize = .06;

iter = @(X,Om) X * expm(Om);
grad = @(X) X * 0.5 * ((X'*A*X*B) - (X'*A*X*B)');
norm = @(X) trace(X'*X);

%% Eigenvalue Decomposition
tic
trial = 0;
trial_limit = 1000;
dist = [];

X = X0;

while trial < trial_limit
    trial = trial + 1;
    
    step = stepSize * X' * grad(X);
    X_new = iter(X,-step);
    dist = [dist,norm(X_new-X)];

    if norm(X_new-X) < 1e-15
        X = X_new;
        break;
    end
    
    X = X_new;
end

toc

%% Result
plot(1:trial,dist);
xlabel('Trials');
ylabel('$||X_n-X_{n-1}||^2$',Interpreter='latex');
title('Convergence of Algorithm');
set(gca,'YScale','log');
set(gcf,'color','white');
fontname('Times New Roman');

disp("A =");
disp(A);
disp('X^T * A * X = ');
disp(X'*A*X);