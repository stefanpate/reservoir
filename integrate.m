% Integrate DDE Mackey Glass and save to .mat file

dt=0.01; T=40; tint=0:dt:T;
tint = tint(1:end-1);
n_samples = 500;
tau = 2;
lags = [tau];

time_series = [];
for j=1:n_samples
    fprintf('%i\n', j)
    sol = dde23(@ddefun, lags, @history, tint);
    y = deval(sol, tint);
    time_series = [time_series; y];
end

% writematrix(time_series, '/nadata/cnl/home/spate/Res/targets/mackey_glass_beta_2_gamma_1_n_9.65_tau_2_n_samples_500_n_steps_4000_dt_0.01.csv');
save('mackey_glass_beta_2_gamma_1_n_9.65_tau_2_n_samples_500_n_steps_4000_dt_0.01.mat', 'time_series');

% sol = dde23(@ddefun, lags, @history, tint);
% y = deval(sol, tint);
% plot(tint, y)

function dydt = ddefun(t,y,Z)
    b = 2; gamma = 1; n=9.65;
    ylag1 = Z(:,1);
    dydt = [b * (ylag1(1) / (1 + ylag1(1)^n)) - gamma * y(1)];
end

function s = history(t)
    s = rand(1,1);
end