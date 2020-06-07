% Angel Villa

clc; clear;

train_file = "USPS_train.txt";

% import and scale input data to [0,1]
train_data = importdata(train_file);
x_train = train_data(:,1:end-1)/max(max(train_data(:,1:end-1)));
y_train = train_data(:,end);

% N: # of training samples
% D: # of input dimensions, not including bias term
% K: # of output dimensions
K = length(unique(y_train));
N = size(x_train,1);
D = size(x_train,2);

% save validation errors and weight matrices
num_val_sets = 10;
validation_error = zeros(num_val_sets,1);
weight_matrices = [];

fileID = fopen('error.txt','w');

% L: # of layers, including input and output layers
% Nh: # of perceptrons in hidden layers
% Nit: # of iterations for training
for L = [3 5 8]
    for Nh = [3 5 8]
        for Nit = [1 15 25]
            for val_ind = 1:num_val_sets
                % calculate validation set start and end indices
                val_start = ceil((val_ind-1)*size(x_train)/num_val_sets + 0.001);
                val_end = floor(val_ind*size(x_train)/num_val_sets);

                % split data in (x,y), (x_test, y_test) = training set, testing set
                x = x_train;
                y = y_train;

                x(val_start:val_end,:) = [];
                y(val_start:val_end,:) = [];

                x_test = x_train(val_start:val_end,:);
                y_test = y_train(val_start:val_end,1);

                % update N each validation cycle
                % U: # of units, including bias unit in input layer
                N = size(x,1);
                U = (L - 2)*Nh + K + D + 1;

                % w: weights for layers l = 2:L
                % w(:,1) weights correspond to bias term for each unit
                % w(i,j) for j>1 is weight connecting unit j from previous layer to unit i
                % i.e. for l=2, j in 1:D+1, for l>2, j in 1:Nh+1
                % rng(1)
                w = 0.1*rand([U-(D+1) max(D+1,Nh+1)]) - 0.05;

                % layers shows the structure of the NN
                % column i -> layer i
                % each row in column corresponds to the index of a unit in that layer
                % 0 if no unit
                layers = zeros(max([D+1,Nh,K]),L);
                layers(1:D+1,1) = (1:D+1)';
                for l=2:L-1
                    layers(1:Nh,l) = ((D+2)+Nh*(l-2):((D+1)+Nh*(l-1)))';
                end
                layers(1:K,end) = (U+1-K:U)';

                % Training
                for it=1:Nit
                    % test other eta values
                    eta = 0.99^(it-1);
                    for n=1:N
                        z = zeros(U,1);
                        a = zeros(U,1);
                        d = zeros(U,1);
                        z(1:(D+1),1) = [1 x(n,:)]';

                        % Calculating 'a' and 'z' for the first hidden layer
                        for i=D+2:D+1+Nh
                            a(i,1) = w(i-(D+1),:)*z(1:(D+1),1);
                            z(i,1) = 1/(1 + exp(-a(i,1)));
                        end

                        % Calculating 'a' and 'z' for the rest of the layers
                        for i=D+2+Nh:U
                            % c_l = current layer
                            % p_i = previous layer indices
                            c_l = floor(find(layers==i)/(D+1)) + 1;
                            p_i = nonzeros(layers(:,c_l-1));
                            a(i,1) = w(i-(D+1),1:Nh+1)*[1; z(min(p_i):max(p_i),1)];
                            z(i,1) = 1/(1 + exp(-a(i,1)));
                        end

                        % Calculating deltas for the output layer
                        j = 0;
                        for i=U-K+1:U
                            if j==y(n)
                                t = 1;
                            else
                                t = 0;
                            end
                            %[j, y(n), t]
                            d(i,1) = (z(i,1)-t)*z(i,1)*(1-z(i,1));
                            j = j + 1;
                        end

                        % Calculating deltas for hidden layers
                        k = Nh + 1;
                        for i=U-K:-1:D+2
                            % c_l = current layer
                            % n_i = next layer indices
                            c_l = floor(find(layers==i)/(D+1)) + 1;
                            n_i = nonzeros(layers(:,c_l+1));
                            sum = 0;
                            for j=min(n_i):max(n_i)
                                sum = sum + d(j,1)*w(j-(D+1),k);
                            end
                            d(i,1) = sum*z(i,1)*(1-z(i,1));
                            k = k - 1;
                            if k == 1
                                k = Nh + 1;
                            end
                        end

                        % Updating weights for layer 2
                        for i=D+2:D+1+Nh
                            for j=1:D+1
                                w(i-(D+1),j) = w(i-(D+1),j) - eta*d(i,1)*z(j,1);
                            end
                        end

                        % Updating weights for layers 3:L
                        for i=D+2+Nh:U
                            % c_l = current layer
                            % p_i = previous layer indices
                            c_l = floor(find(layers==i)/(D+1)) + 1;
                            p_i = nonzeros(layers(:,c_l-1));
                            w(i-(D+1),1) = w(i-(D+1),1) - eta*d(i,1);
                            k = 2;
                            for j=min(p_i):max(p_i)
                                w(i-(D+1),k) = w(i-(D+1),k) - eta*d(i,1)*z(j,1);
                                k = k + 1;
                            end
                        end
                    end
                end

                % Testing
                N_test = size(x_test,1);
                correct = 0;
                for n=1:N_test
                    z = zeros(U,1);
                    a = zeros(U,1);
                    z(1:(D+1),1) = [1 x_test(n,:)]';

                    % Calculating 'a' and 'z' for the first hidden layer
                    for i=D+2:D+1+Nh
                        a(i,1) = w(i-(D+1),:)*z(1:(D+1),1);
                        z(i,1) = 1/(1 + exp(-a(i,1)));
                    end

                    % Calculating 'a' and 'z' for the rest of the layers
                    for i=D+2+Nh:U
                        % c_l = current layer
                        % p_i = previous layer indices
                        c_l = floor(find(layers==i)/(D+1)) + 1;
                        p_i = nonzeros(layers(:,c_l-1));
                        a(i,1) = w(i-(D+1),1:Nh+1)*[1; z(min(p_i):max(p_i),1)];
                        z(i,1) = 1/(1 + exp(-a(i,1)));
                    end    

                    % p_c: predicted class
                    % digits in [0, 1, . . . , 9] but
                    % p_c in [1, 2, . . . , 10] so
                    % subtract 1 to get actual predicted class, need to
                    % tweak depending on the data set
                    [val p_c] = max(z(U-K+1:U,1));
                    p_c = p_c - 1;

                    %fprintf('Datum ID = %d, Predicted Class = %d, True Class = %d \n', n, p_c, y_test(n,1));
                    if y_test(n,1) == p_c
                        correct = correct + 1;
                    end
                end
                validation_error(val_ind) = 1 - correct/N_test;
                weight_matrices = [weight_matrices; val_ind*ones(1,size(w,2)); w];
            end
            fprintf(fileID,'\nLayers: %d, Hidden units: %d, Iterations: %d \n',L,Nh,Nit);
            fprintf(fileID,'Validation error: ');
            fprintf(fileID,'%0.3f, ',validation_error);
        end
    end
end