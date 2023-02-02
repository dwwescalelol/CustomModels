classdef kd_ClassificationKNN < handle
    %KD_CLASSIFICATIONKNN K Nearest Neighbors classification with K-D Tree
    % KD_CLASSIFICATIONKNN is a K Nearest Neighbors (KNN) classification model.
    % It can predict response for new data. 
    %
    % KD_CLASSIFICATIONKNN Properties:
    %    X              - Training examples.
    %    Y              - Training labels.
    %    kd             - K-D tree of training examples.
    %    NumNeighbors   - Number of nearest neighbors.
    %    Verbose        - Control for debug.
    %
    % KD_CLASSIFICATIONKNN Methods:
    %    predict        - Returns a matrix of predicted classifications for each
    %                     observation (number of observations, 1).
    %    knnsearch      - Returns the nearest NumNeigbor indexs to input observation.
    %    calcED         - Calculates the Euclidian Distance between two observations.

    properties              
        X               % - Training examples.
        Y               % - Training labels.
        kd kd_Tree      % - kd tree of training examples.
        NumNeighbors    % - Number of nearest neighbors. 
        Verbose         % - Control for debug.
        TreeSearched    % - Number of nodes in tree searched.
    end
    
    methods
                
        function obj = kd_ClassificationKNN(X, Y, NumNeighbors, Verbose)            
            %Constructs an instance of this class.
            % set up our training data:
            obj.X = X;
            obj.Y = Y;

            % construct kd tree
            obj.kd = kd_Tree(obj.X);

            % store the number of nearest neighbours we're using:
            obj.NumNeighbors = NumNeighbors;
            
            % are we printing out debug as we go?
            obj.Verbose = Verbose;         
        end
                
        function predictions = predict(obj, test_examples)          
            %PREDICT Returns a (number of observations, 1) matrix of predicted classifications for each observation.

            % initialise array so that rows = test data and coloumns = k neighbours
            knnIndexs = zeros(height(test_examples),obj.NumNeighbors);

            % fknnIndexs knn for each obs in test, add to knnIndexs
            for i=1:height(test_examples)
                knnIndexs(i,:) = knnsearch(obj,test_examples(i,:));                   
            end

            % find lable of each index
            predictions = obj.Y(knnIndexs);
            predictions = mode(predictions,2);
        end

        % Finds K nearest points in X from test_obj 
        function knnIndexs = knnsearch(obj, test_obj)
            %KNNSEARCH Find K nearest neighbours.           
            % IDX = KNNSEARCH(test_obj) finds the nearest k neighbours in obj.X to
            % test_obs. IDX is a k by 1 matrix.

            % make sure test_obs is same format as training data.
            if width(test_obj) ~= width(obj.X)
                error("Current point and training exampels do not have the same ammount of features.")
            end

            % knnIndexs = [index,distance] where distance is ed from test_obj
            knnIndexs = Inf(obj.NumNeighbors,2);

            % find knn
            knnIndexs = findnn(obj, obj.kd.Root, knnIndexs, test_obj, 1);

            % IDX should only include index coloumn of indexs, distances not important.            
            knnIndexs = knnIndexs(:,1);

            if obj.Verbose
                obj.TreeSearched(end + 1) = obj.kd.SearchedNodes./length(obj.X) .* 100;
                fprintf("%% of tree: %f\n", obj.kd.SearchedNodes./length(obj.X) .* 100)
                obj.kd.SearchedNodes = 0;
            end
        end
        
        function values = findnn(obj, node, values, obs, dim)
            %FINDNN recursivly searches for the nearest k neighbours of a
            %given observation.
            % values = FINDNN(node, values, obs, dim) where node is the
            % current node, values is the persisting data containing both indexs 
            % of training data and distances of each index from observation,
            % obs is observation to find knn for and dim is current
            % dimention being assessed. Values is in format [indexs, distances]

            % break condition 
            if isempty(node)
                return
            end    

            if obj.Verbose
                obj.kd.incnodesvisited();
            end

            % decide order of children to be searched
            if obs(dim) < node.Obs(dim)
                nextNode = node.Left;
                delayedNode = node.Right;
            else
                nextNode = node.Right;
                delayedNode = node.Left;
            end
            
            % if nextNode not empty, recurse down to best fitting limit
            if ~isempty(nextNode)
                values = findnn(obj, nextNode, values, obs, mod(dim, obj.kd.Dimentions) + 1);
            end

            % reassign if dist is less than max of k current closest
            dist = sum(((obs - node.Obs).^2));

            % if tie for max distances, choses distance with largest index by default
            % [maxValue,maxInd] = obj.mintrainind(values);
            [maxValue,maxInd] = max(values(:,2));
            if dist < maxValue || (dist == maxValue && node.Index < values(maxInd,1))
                values(maxInd,:) = [node.Index, sum(((obs - node.Obs).^2))];
                % sort so that highest index is favoured
                values = sortrows(values,1,'descend');
            end                
            
            % if delayedNode not empty, find shortest distance from point to hplane
            if ~isempty(delayedNode)
                % if distance < max nearest neighbour, search, recurse down
                if shortestdistance(obj, obs, delayedNode) <= maxValue
                    values = findnn(obj, delayedNode, values, obs, mod(dim, obj.kd.Dimentions) + 1);
                end                
            end

        end

        function shortestDistance = shortestdistance(obj, point, node)
            %SHORTESTDISTANCE calculates the shortest distnace between a
            %point and a hyperplane.
            % shortestDistance = SHORTESTDISTANCE(point, node) where the
            % node contains the hyperplane. 

            if isempty(node)
                error("kd_Node not passed in as argumentr")
            end
            
            limits = node.Limits;
            closestPoint = zeros(1,size(point,2));

            % point lies in limits use point, high or lower bound if not
            for i=1:width(point)                
                if point(i) >= limits(i,1) && point(i) <= limits(i,2)
                    closestPoint(i) = point(i);
                elseif point(i) < limits(i,1)                  
                    closestPoint(i) = limits(i,1);
                else                    
                    closestPoint(i) = limits(i,2);
                end
            end
            % calc distance from closestPoint to point
            shortestDistance = sum(((point - closestPoint).^2));
        end
    end

    methods(Static)
        % values is a matrix of [index,distance]  
        function [maxValue, maxInd] = mintrainind(values)
            %MINTRAININD Finds the max value of given k by 2 matrix
            % [maxValue, maxInd] = mintrainind where maxInd is the index in
            % values which has the largest value. If multiple instances of
            % maxValue index with minimum value in coloumn 1 is chosen
            % (smallest index in training data).

            % find max distance
            maxValue = max(values(:,2));
            % find where max distance occours in values
            maxInd = values(:,2) == maxValue;
            % find index in values of smallest training index with max value
            maxInd = min(values(maxInd,1));
            % find the index in values
            maxInd = find(values(:,1) == maxInd);
        end
    end    
end