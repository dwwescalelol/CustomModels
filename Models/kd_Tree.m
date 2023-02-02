classdef kd_Tree < handle
    %kd_Tree A tree to store training data of K dimentions in for faster searches.
    % Reduces search time average fro O(N) to O(log(N)).
    % Author: Zaz
    %
    % kd_Tree Properties:
    %    root           - Root node of tree.
    %    dimentions     - Number of dimentions in the given dataset.
    %    searchedNodes  - Number of nodes searched.
    %
    % kd_Tree Methods:
    %   makeNodes       - Recursively constructs a tree from a given values.
    %   findAxis        - Finds what axis should represent a given height of the tree.
    %   incNodesVisited - Increments the searched nodes count by 1.
    %   calcMedianInd   - Finds the median index of an matrix's rows.

    properties
        Root            % Root node of tree.
        Dimentions      % Number of axis in the given dataset.
        SearchedNodes   % Number of nodes searched.
    end

    methods

        function obj = kd_Tree(values)
            %kd_Tree Constricts a instance of this class. Values should be 
            %passed so that each training observation is a row and each feature value is a coloumn.

            % Dimentions is number of features
            obj.Dimentions = width(values);
           
            % find limits of data
            trainLimits = obj.calclimits(obj.Dimentions,values);
  
            % preserve indexs of data
            values(:,end + 1) = 1:height(values);
          
            % create tree
            obj.Root = makeNodes(obj,values,1,trainLimits);

            obj.SearchedNodes = 0;
        end

        function node = makeNodes(obj, values, axis, limits)
            %MAKENODES Recursively constructs a tree from a given values.
            % Stops if there are no values to construct a node from.

            if isempty(values)
                node = [];
                return 
            end

            % Select axis based on depth so that axis cycles through all valid values
            values = sortrows(values,axis);
            % index of obs in sorted array, not index of obs in train_exampls
            medianIndex = round(height(values)/2);
            
            nodeObs = values(medianIndex,1:obj.Dimentions);
            % preserve index of obs in training_examples
            obsIndex = values(medianIndex,end);

            % create node with current limits
            node = kd_Node(nodeObs,obsIndex,limits);
                      
            % specify limits of children
            limitsL = limits;
            limitsR = limits;

            limitsL(axis,2) = nodeObs(axis);
            limitsR(axis,1) = nodeObs(axis);
    
            % assign working axis for children
            axis = mod(axis,obj.Dimentions) + 1;

            % recurse in children
            node.Left = makeNodes(obj,values(1:medianIndex - 1,:), axis, limitsL);           
            node.Right = makeNodes(obj,values(medianIndex + 1:end,:), axis, limitsR);
        end

        function incnodesvisited(obj)
            %INCNODESVISITED increments the SearchedNodes property by 1.
            obj.SearchedNodes = obj.SearchedNodes + 1;
        end
    end

    methods(Static)
        function limits = calclimits(dimentions, values)
            %CALCLIMITS calculates the bounds that an observation can
            %exsist in. 
            % limits = calclimits(dimentions, values) where limits is a dimentions
            % by 2 matrix where the coloumns are [min, max] where each row
            % represents a differnt dimention. 

            limits = zeros(dimentions,2);

            for i=1:dimentions
                limits(i,1) = min(values(:,i));
                limits(i,2) = max(values(:,i));
            end
        end
    end
end