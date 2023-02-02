classdef my_ClassificationKNN < my_ClassificationModel
    %my_ClassificationKNN K Nearest Neighbors classification 
    %   my_ClassificationKNN is a K Nearest Neighbors (KNN) classification model.
    %   It can predict response for new data. It also stores data used for
    %   training.

    properties              
        NumNeighbors    % - Number of nearest neighbours to consider.
    end
    
    methods
        function obj = my_ClassificationKNN(X, Y, NumNeighbors)           
            % assign superclass properties
            obj@my_ClassificationModel(X,Y);

            % assign class specific properties
            obj.NumNeighbors = NumNeighbors;            
        end
                
        function [predictions, scores] = predict(obj, test_examples)
            % initialise array so that rows = test data and coloumns = k neighbours
            ind = zeros(height(test_examples),obj.NumNeighbors);
            scores = zeros(height(test_examples),height(obj.ClassNames));
            
            % for each test example find the knn and distribution of classes in those nn
            for i=1:height(test_examples)
                ind(i,:) = obj.my_knnsearch(test_examples(i,:));   

                % log the frequency of each class label at each prediction
                for j = 1:height(obj.ClassNames)
                    labels = obj.Y(ind(i,:));
                    scores(i,j) = length(labels(labels == obj.ClassNames(j)));
                end
            end

            % finds the mode across rows rather than coloumns (takes second axis)
            predictions = mode(obj.Y(ind),2);

            % make so sum of scores(i,:) = 1 as should be probability
            scores = scores / obj.NumNeighbors;
        end        

        % helper function for predict, returns k nearest neighbours for a test obs
        function kClosestIndexs = my_knnsearch(obj, test_obs) 
            % initialise virtical array, same height as no of observations
            kClosestIndexs = zeros(height(obj.X),1);

            % store distance between test observation and each observation
            % in train_examples
            for i=1:length(kClosestIndexs)
                kClosestIndexs(i,1) = sum((test_obs(:,:) - obj.X(i,:)).^2);
            end
            
            % store indexs of k distances from a point to test obs
            [~,kClosestIndexs] = sort(kClosestIndexs);
            kClosestIndexs = kClosestIndexs(1:obj.NumNeighbors);
        end
    end  
end
