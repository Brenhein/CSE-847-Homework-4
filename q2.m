%% Get me data
ad_data = load('C:\Users\brenh\Desktop\SS22\CSE 847\HW4\ad_data.mat');
features = load('C:\Users\brenh\Desktop\SS22\CSE 847\HW4\feature_name.mat');
train_x = ad_data.X_train;
train_y = ad_data.y_train;
test_x = ad_data.X_test;
test_y = ad_data.y_test;
features = features.FeatureNames_PET;

%% Set the options
opts.rFlag = 1; % range of par within [0, 1].
opts.tol = 1e-6; % optimization precision
opts.tFlag = 4; % termination options.
opts.maxIter = 5000; % maximum iterations.

%% Call the model for each
pars = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
features_s = [];
for i = 1:length(pars)
    
    %% Train the model for a given l1 regularization parameter
    par = pars(i);
    [w, c] = LogisticR(train_x, train_y, par, opts);

    %% Find the number of non-zero weights
    cnt = 0;
    for j = 1:length(w)
        if w(j) ~= 0
            cnt = cnt + 1;
        end 
    end 
    features_s(end + 1) = cnt;

    %% Calculate the AUC metrix
    predictions = [];
    for j = 1:length(test_y)
        x = test_x(j, :);
        y = sign(dot(w, x));
        predictions(end + 1) = y;
    end
    [X, Y] = perfcurve(test_y, predictions, 1);
    
    %% Plot a subplot of the current AUC
    subplot(3, 4, i);
    plot(X, Y);
    head = strcat("λ=", string(par));
    title(head);
    xlabel("X");
    ylabel("Y");  
end

%% Save the plot of all the subplots for AUC
name = 'C:\Users\brenh\Desktop\SS22\CSE 847\HW4\auc.jpg';
saveas(gcf, name, 'jpeg');

%% Plot the regularization parameters and number of features selected
figure;
plot(pars, features_s);
title("Regularization Parameter vs The Number of Selected Features")
xlabel("The Regularization Parameter")
ylabel("The Number of Features Selected")
saveas(gcf, ...
     'C:\Users\brenh\Desktop\SS22\CSE 847\HW4\feature_cnt.jpg', ...
     'jpeg');


