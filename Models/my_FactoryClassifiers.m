function m = my_FactoryClassifiers(X, Y, Method, Learners, NumLearningCycles, NFeatureToSample, NObsToSample)
    switch Method
        case 'SoftVote'
            m = makeSoftVoter(X, Y, Learners);
        case 'Bag'
            m = makeBag(X, Y, Learners, NumLearningCycles, NObsToSample);
        case 'Subspace'
            m = makeSubspace(X, Y, Learners, NumLearningCycles, NFeatureToSample);
        case 'RandomForest'
            m = makeRandomForest(X, Y, Learners, NumLearningCycles, NFeatureToSample, NObsToSample);
        otherwise
            error("Method argument not defined, choises are " + ...
                "SoftVote, Bag, Subspace and RandomForest.")
    end
end


function m = makeSoftVoter(X, Y, learners)
    %makeSoftVoter makes a soft voter where number of classifiers made is 
    % same as templates supplied. Returns cell array of classifiers. Tried 
    % to use homogenious arrays but Matlab has limited support for 
    % polymorphrism. Used as a factory in ensamble call.

    % wrapped so learners can be cell indexed even when one element
    if length(learners) == 1
        learners = {learners};
    end

    m = cell(length(learners),1);
    for i=1:height(m)
        m{i} = makeClassifier(X, Y, learners{i});
    end
end

function m = makeBag(X, Y, learners, numLearningCycles, nObsToSample)
    %makeBag random observations 


end

function m = makeSubspace(X, Y, learners, nFeatureToSample);


end

function m = makeRandomForest(X, Y, learners, nFeatureToSample, nObsToSample)


end

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