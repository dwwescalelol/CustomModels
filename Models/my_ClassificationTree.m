classdef my_ClassificationTree < my_ClassificationModel
    %my_ClassificationTree Decision tree for classification.
    %   ClassificationTree is a decision tree with binary splits for
    %   classification. It can predict response for new data. It also stores
    %   data used for training.

    properties
        CutPredictorIndex   % - Index vector for the split predictors for tree nodes.
        CutPoint            % - Points for splits on continuous predictors.
        Children            % - Child nodes for tree nodes.
        ClassProbability    % - Class probabilities for tree nodes.
        NodeClass           % - Majority class per tree node.
        NodeRisk            % - WGDI of node.
        NumNodes            % - Number of nodes in tree.
        MinParentSize       % - Minimum amount of data to be a parent.
        MaxNumSplits        % - Maximum total splits in tree.
    end
   
    methods
        function obj = my_ClassificationTree(X, Y, MinParentSize, MaxNumSplits)
            % call superclass
            obj@my_ClassificationModel(X,Y);
            
            % tree acts as a factory
            tree = my_DecisionTree(X, Y, MinParentSize, MaxNumSplits);

            % asssign class properties
            obj.MinParentSize = MinParentSize;
            obj.MaxNumSplits = MaxNumSplits;
            obj.CutPredictorIndex = tree.CutPredictorIndex;
            obj.CutPoint = tree.CutPoint;
            obj.ClassProbability = tree.ClassProbability;
            obj.Children = tree.Children;
            obj.NodeClass = tree.NodeClass;
            obj.NodeRisk = tree.NodeRisk;
            obj.NumNodes = tree.NumNodes;
        end

        function [predictions, scores] = predict(obj, test_examples)           
            % get ready to store our predicted class labels:
            indNode = zeros(height(test_examples),1);

            for i=1:height(test_examples)
                % initialise obs and index
                nodeIndex = 1;
                obs = test_examples(i,:);

                % keep descending until we reach a leaf node
                while (obj.Children(nodeIndex, 1) ~= 0)
                    targetVal = obs(obj.CutPredictorIndex(nodeIndex));
                
                    if targetVal < obj.CutPoint(nodeIndex)
                        nodeIndex = obj.Children(nodeIndex, 1);
                    else 
                        nodeIndex = obj.Children(nodeIndex, 2);
                    end
                end
                % when leaf assign prediction to label of leaf
                indNode(i) = nodeIndex;
            end
            predictions = obj.NodeClass(indNode);
            scores = obj.ClassProbability(indNode,:);
        end

        % matlabs ClassificationTree has a view method, is not separate function.
        function view(obj)
            % format nodes for diagraph function
            nodes = find(obj.Children(:,1)~=0)';
            % duplicates all elements in matrix
            nodes = repelem(nodes, 2);
            % assign root at start of matrix
            nodes = [0 nodes];
        
            % so that diagram has different text for leaves and parents.
            nodeText = cell(1,length(nodes));
            isParent = obj.Children(:,1)~=0;
            for i=1:height(isParent)
                if isParent(i)
                    nodeText{i} = "x" + obj.CutPredictorIndex(i) + " : " + obj.CutPoint(i);
                else
                    nodeText{i} = string(obj.NodeClass(i));
                end
                nodeText{i} = char(nodeText{i});
            end
        
            % plot graph
            figure;
            g = digraph(nodes(nodes~=0),find(nodes));    
            plot(g,'ShowArrows',false,'NodeLabel',nodeText,'NodeFontSize',11);
        end
    end
end