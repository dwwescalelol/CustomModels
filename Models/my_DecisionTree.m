classdef my_DecisionTree < handle
    %my_DecisionTree makes a decision tree. Is used as a factory in the
    %   my_ClassificationTree class. Does not use recursion for reasons 
    %   explored in the Extention Work folder.

    properties
        Data                % - Data at each node.
        Labels              % - Labels at each node.
        MinParentSize       % - Minimum parent node size.
        MaxNumSplits        % - Maximum number of splits.      
        CutPredictorIndex   % - Index vector for the split predictors for tree nodes.
        CutPoint            % - Points for splits on continuous predictors.
        ClassProbability    % - Class probabilities for tree nodes.
        Children            % - Child nodes for tree nodes.
        NodeClass           % - Majority class per tree node.
        NodeRisk            % - WGDI of node.
        ClassNames          % - All unique class names.
        NumNodes            % - Stores number of nodes as tree is constructed.
        NumSplits           % - Stores number of splits as tree is constructed.
        Layer               % - Layer of node. Used for MaxNumSplits.
    end

    methods
        function obj = my_DecisionTree(X, Y, MinParentSize, MaxNumSplits)
            % set up our training data:
            obj.Data{1} = X;
            obj.Labels{1} = Y;
            
            % store the minimum parent node size we're using:
            obj.MinParentSize = MinParentSize;

            % for splits hyperparam
            obj.MaxNumSplits = MaxNumSplits;
            obj.NumSplits = 0;

            % only node is root node
            obj.NumNodes = 1;

            % initialise root valuescreateParentNode
            obj.ClassNames = unique(Y);            
            obj.NodeRisk = obj.calcwgdi(Y);
            obj.NodeClass = mode(obj.Labels{1});
            obj.Children = [0,0];
            obj.ClassProbability = obj.calcClassProbability(Y);
            obj.Layer = 1;
            obj.createTree();
        end

        function createTree(obj)
            currentNode = 1;
            % sum is total nodes currently in scope
            while currentNode <= obj.NumNodes  && obj.NumSplits < obj.MaxNumSplits
                % must have more data than min parent to become parent
                parentSize = height(obj.Labels{currentNode});
                if parentSize >= obj.MinParentSize
                    obj.createNode(currentNode);
                end
                % evaluate next node
                currentNode = currentNode + 1;
            end

            if obj.NumSplits >= obj.MaxNumSplits
                obj.configureBottomLayer();
            end
        end
        
        function createNode(obj, currentNode)
            % assign node properties
            obj.NodeClass(currentNode,1) = mode(obj.Labels{currentNode});
            % find splits of children
            [split, feature, cutPoint, wgdi] = obj.findsplit(obj.Data{currentNode}, obj.Labels{currentNode});
    
            % if current node WGDI > sum children WGDI and 
            if sum(wgdi) < obj.NodeRisk(currentNode)
                % make node
                obj.createParentNode(split, feature, cutPoint, wgdi, currentNode);
                % increment properties
                obj.NumNodes = obj.NumNodes + 2;
                obj.NumSplits = obj.NumSplits + 1;
            end
        end

        % makes node data and assigns values to appropriate property matrixs
        % should be noted that when a parent is constructed children are
        % also made to avoid recalculating wgdi in future.
        function createParentNode(obj, split, feature, cutPoint, wgdi, currentNode)
            % get children indexs
            leftIndex = obj.NumNodes + 1;
            rightIndex = leftIndex + 1;

            % assign current node properties
            obj.CutPoint(currentNode,1) = cutPoint;
            obj.Children(currentNode,:) = [leftIndex,rightIndex];
            obj.CutPredictorIndex(currentNode,1) = feature;
    
            % split data for children
            [rightData, rightLabels, leftData, leftLabels] = obj.splitdata(split,feature,obj.Data{currentNode},obj.Labels{currentNode});
    
            % construct children
            nextLayer = obj.Layer(currentNode) + 1;

            obj.createChildNode(leftIndex, leftData, leftLabels, wgdi(1), nextLayer);
            obj.createChildNode(rightIndex, rightData, rightLabels, wgdi(2), nextLayer);
        end

        function createChildNode(obj, node, data, labels, wgdi, layer)
            % initialises all values to default for child node
            obj.CutPoint(node,1) = NaN;
            obj.Children(node,:) = [0 0];
            obj.CutPredictorIndex(node,1) = 0;

            % assign values to leafe node
            obj.Data{node} = data;
            obj.Labels{node} = labels;
            obj.NodeRisk(node,1) = wgdi;
            obj.NodeClass(node,1) = mode(labels);
            obj.ClassProbability(node,:) = obj.calcClassProbability(labels);
            obj.Layer(node,1) = layer;
        end

        % takes sorted feature values. feature value is not needed.
        % splits input labels at all indexs and finds best WGDI of both.
        function [split, feature, cutPoint, bestwgdi] = findsplit(obj, data, labels)
            % initalise all values so that if imidietly returned does not crash
            bestwgdi = Inf(1,2);
            currentwgdi = Inf(1,2);
            split = 0;
            feature = 1;
            cutPoint = 0;

            for i=1:width(data)
                % sort data by each feature, sort labels by the same
                [data,indexs] = sortrows(data,i);
                labels = labels(indexs);

                % used to skip duplicate values so that splits are on first
                % instance of a value.
                lastResult = data(1,i);
               
                % foreach unique feature find wgdi
                for j=2:height(data)
                    % only search unique indexs, skip all duplicates
                    if data(j,i) == lastResult
                        continue
                    end

                    lastResult = data(j,i);
                    leftLabels = labels(1:j - 1);
                    rightLabels = labels(j:end);

                    % left and right wgdis
                    currentwgdi(1) = obj.calcwgdi(leftLabels) ;
                    currentwgdi(2) = obj.calcwgdi(rightLabels);

                    % if better wgdi found reassign
                    if sum(currentwgdi) < sum(bestwgdi)
                        bestwgdi = currentwgdi;
                        split = j;
                        feature = i;
                        cutPoint = (data(j-1,feature) + data(j,feature))./2;
                   end
                end
            end
        end

        function classProbabilities = calcClassProbability(obj, labels)
            classProbabilities = zeros(1,height(obj.ClassNames));

            % foreach class store the frequency of that class in labels
            for i=1:height(obj.ClassNames)
                classFreq = length(labels(labels == obj.ClassNames(i)));
                
                % devide by ammount of labels to get distribution
                classProbabilities(i) = classFreq / height(labels);
            end
        end

        % used insted of Jhon's given function
        function wgdi = calcwgdi(obj, labels)
            % make matrix of numClassNames by 1
            labelDistribution = zeros(height(obj.ClassNames),1);

            % find weight
            numObservations = height(obj.Labels{1});
            weight = height(labels)/numObservations;

            % store distribution of classes in labels
            for i=1:height(obj.ClassNames)
                freqOfClass = length(labels(labels== obj.ClassNames(i)));
                labelDistribution(i) =  freqOfClass / height(labels);
            end

            % gdi = 1 - sum(%of class ^2)
            wgdi = (1 - sum(labelDistribution.^2)) * weight;
        end

        function makeLeafNode(obj, node)
            % initialises all values to default for leaf node
            obj.CutPoint(node,1) = NaN;
            obj.Children(node,:) = [0 0];
            obj.CutPredictorIndex(node,1) = 0;
        end

        % finds best nodes at bottom layer to split and splits them
        % only called if numSplits >= MaxNumSpltis
        function configureBottomLayer(obj)
            % have to find the parent nodes at the bottom, second last layer.
            parentLayer = max(obj.Layer - 1);
            bottomPNodes = find(obj.Layer == parentLayer);
            
            % make all nodes on second to last row leaves
            for i=1:height(bottomPNodes)
                obj.makeLeafNode(bottomPNodes(i))
            end

            % find amount of leaves on bottom row, this is how many
            % children need to be made.
            childrenToMake = height(find(obj.Layer == max(obj.Layer)));
            % numnodes used to track index of next node to be made, needs
            % to be reset.
            obj.NumNodes = obj.NumNodes - childrenToMake;
            splitsToDo = (childrenToMake / 2);
            

            % 2 to store bestwgdi to make new parent
            nodeData = [bottomPNodes zeros(height(bottomPNodes), 2)];
            % will find ideal split via sorting deltaWGDIs 
            deltaWGDI = zeros(height(bottomPNodes),1);

            % assign deltaWGDI 
            for i=1:height(nodeData)
                [~, ~, ~, bestwgdi] = obj.findsplit(obj.Data{bottomPNodes(i)}, obj.Labels{bottomPNodes(i)});
                nodeData(i,2:end) = bestwgdi;
                deltaWGDI(i) = obj.NodeRisk(bottomPNodes(i)) - sum(bestwgdi);
            end

            % find indexs of deltaWGDI sorted from highest to lowest 
            [~, indexDWGDI] = sort(deltaWGDI,'descend');
            % only make parents that have best deltaWGDI
            indexDWGDI = indexDWGDI(1: splitsToDo);
            % sort list so that bottomPNodes are created in order
            parentData = sortrows(nodeData(indexDWGDI,:),1);

            % create bottomPNodes 
            for i=1:height(parentData)
                node = parentData(i,1);
                obj.createNode(node);
            end
        end
    end
    methods (Static)
        function [rightData, rightLabels, leftData, leftLabels] = splitdata(split, feature, data, labels)
            % sort data and labels by correct feature
            [data,indexs] = sortrows(data,feature);
            labels = labels(indexs);
            
            % left contains strictly less than split values
            leftData = data(1:split - 1,:);
            leftLabels = labels(1:split - 1,:);
            
            % right contains greater than or equal to
            rightData = data(split: end,:);
            rightLabels = labels(split: end,:);
        end
    end
end