classdef my_ClassifierFactory < handle
    %my_ClassifierFactory is a factory class for the
    % my_ClassificationEnsamble class, can handel all types of ensambles
    % convered on this course. (Although it may take a very long time :))

    properties
        X                   % - Training Data.
        Y                   % - Training Lables.
        Method              % - Learner aggregation method.
        Learners            % - Cell array of my_template objects.
        NPredToSample       % - Number of features to account for in bagging and random forest.
        NumLearningCycles   % - Number of classifiers to train (excludes soft voting).
    end

    methods
        function obj = my_ClassifierFactory(X, Y, Method, Learners, NumLearningCycles, NPredToSample)
            obj.X = X;
            obj.Y = Y;
            obj.Method = Method;
            obj.Learners = Learners;
            obj.NumLearningCycles = NumLearningCycles;
            obj.NPredToSample = NPredToSample;

            % wrapped so learners can be cell indexed even when one element
            if length(Learners) == 1
                obj.Learners = {obj.Learners};
            end
        end
        
        function m = makeClassifiers(obj)
            switch obj.Method
                case 'SoftVote'
                    m = obj.makeSoftVoter();
                case 'Bag'
                    m = obj.makeBag();
                case 'Subspace'
                    m = obj.makeSubspace();
                case 'RandomForest'
                    m = obj.makeRandomForest();
                otherwise
                    error("Method argument not defined, choises are " + ...
                        "SoftVote, Bag, Subspace and RandomForest.")
            end
        end
        
        
        function m = makeSoftVoter(obj)
            %makeSoftVoter makes a soft voter where number of classifiers made is 
            % same as templates supplied. Returns cell array of classifiers. Tried 
            % to use homogenious arrays but Matlab has limited support for 
            % polymorphrism. Used as a factory in ensamble call.
                
            m = cell(length(obj.Learners),1);
            for i=1:height(m)
                m{i} = obj.makeClassifier(obj.X, obj.Y, obj.Learners{i});
            end
        end
        
        function m = makeBag(obj)
            %makeBag random observations 
            numTemplates = length(obj.Learners);
            m = cell(obj.NumLearningCycles * numTemplates,1);
            
            % used for rand function
            numObs = height(obj.X);

            % for each template for NumLearnCycles, make classifier.
            for i=1:numTemplates:obj.NumLearningCycles * numTemplates
                obsIndexs = randi(numObs,numObs,1);

                for j=1:numTemplates
                    m{i + j - 1} = obj.makeClassifier(obj.X(obsIndexs,:), ...
                        obj.Y(obsIndexs), obj.Learners{j});
                end
            end
        
        end
        
        function m = makeSubspace(obj)
            %makeBag random observations 
            numTemplates = length(obj.Learners);
            m = cell(obj.NumLearningCycles * numTemplates,1);
            
            % used for rand function
            numFeatures = width(obj.X);

            % for each template for NumLearnCycles, make classifier.
            for i=1:numTemplates:obj.NumLearningCycles * numTemplates
                predictorIndexs = randperm(numFeatures);
                predictorIndexs = predictorIndexs(1:obj.NPredToSample);
                predictorIndexs = sort(predictorIndexs);
                % data with only nPredToSample random features
                x = obj.X(:,predictorIndexs);

                for j=1:numTemplates
                     c = obj.makeClassifier( ...
                        x, obj.Y, obj.Learners{j});
                     % so that predictions will be made with the correct features
                     c.PredictorIndexs = predictorIndexs;

                     m{i + j - 1} = c;
                end
            end
        
        end
        
        function m = makeRandomForest(obj)
            %makeBag random observations 
            numTemplates = length(obj.Learners);
            m = cell(obj.NumLearningCycles * numTemplates,1);
            
            % used for rand function
            numFeatures = width(obj.X);
            numObs = height(obj.X);

            % for each template for NumLearnCycles, make classifier.
            for i=1:numTemplates:obj.NumLearningCycles * numTemplates
                % get indexs to order train examples
                obsIndexs = randi(numObs,numObs,1);
                
                predictorIndexs = randperm(numFeatures);
                predictorIndexs = predictorIndexs(1:obj.NPredToSample);
                predictorIndexs = sort(predictorIndexs);
                % data with only nPredToSample random features
                x = obj.X(obsIndexs,predictorIndexs);

                for j=1:numTemplates
                     c = obj.makeClassifier(x, obj.Y, obj.Learners{j});
                     % so that predictions will be made with the correct features
                     c.PredictorIndexs = predictorIndexs;

                     m{i + j - 1} = c;
                end
            end
            
        end
        
    end
    methods (Static)
        function m = makeClassifier(X, Y, template)
            %makeClassifier takes a template and creates a classifier with the
            %   given X and Y values. Used for ensasmbles.
            
            % switch case is based of template type
            modelType = class(template);
        
            switch modelType
                case 'my_templateKNN'
                    m = my_ClassificationKNN(X, Y, template.NumNeighbors);
        
                case 'my_templateTree'
                    % MaxNumSplits can only set default value when training data is
                    % specified. MaxNumSplits set to -1 in template when no value given
                    if template.MaxNumSplits == -1
                       template.MaxNumSplits = height(X) - 1;
                    end
                    
                    m = my_ClassificationTree(X, Y, ...
                        template.MinParentSize, template.MaxNumSplits);
        
                case 'my_templateNB'
                    m = my_ClassificationNaiveBayes(X, Y);
        
                otherwise
                    error("Template in learners hyperparameter is not a defined my_template class.")
            end
        end
    end
end