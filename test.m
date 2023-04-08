% cargar el archivo de csv a normalito
%filename = 'breast-cancer-wisconsin.data.csv'
%datos = readtable(filename)
%datos = csvread(filename);
% separar benignos y malignos

% Cargar los datos
data = readtable('breast-cancer-wisconsin.data', 'Format', '%f%f%f%f%f%f%f%f%f%f%f');

% Reemplazar los valores faltantes con la media
data = standardizeMissing(data, '?');
data = fillmissing(data, 'movmean', 2);

% Dividir los datos en conjuntos de entrenamiento y prueba
X = data(:, 2:10);
y = data(:, 11);
cv = cvpartition(height(data), 'HoldOut', 0.2);
X_train = X(training(cv), :);
X_test = X(test(cv), :);
y_train = y(training(cv), :);
y_test = y(test(cv), :);

% Entrenar el modelo
priors = [sum(y_train.Var11 == 2) / height(y_train); sum(y_train.Var11 == 4) / height(y_train)];
likelihoods = zeros(2, width(X_train));
for i = 1:width(X_train)
    likelihoods(1, i) = mean(X_train{y_train.Var11 == 2, i});
    likelihoods(2, i) = mean(X_train{y_train.Var11 == 4, i});
end

% Hacer predicciones en el conjunto de prueba
y_pred = zeros(height(y_test), 1);
for i = 1:height(X_test)
    posterior_2 = priors(1);
    posterior_4 = priors(2);
    for j = 1:width(X_test)
        posterior_2 = posterior_2 * normpdf(X_test{i, j}, likelihoods(1, j), std(X_train{y_train.Var11 == 2, j}));
        posterior_4 = posterior_4 * normpdf(X_test{i, j}, likelihoods(2, j), std(X_train{y_train.Var11 == 4, j}));
    end
    if posterior_2 > posterior_4
        y_pred(i) = 2;
    else
        y_pred(i) = 4;
    end
end

% Calcular la precisión del modelo
accuracy = sum(y_pred == y_test.Var11) / height(y_test);
disp(['Accuracy: ', num2str(accuracy)])
