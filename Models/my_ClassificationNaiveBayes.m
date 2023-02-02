classdef my_ClassificationNaiveBayes < my_ClassificationModel
   
    properties
        Prior                   % the prior probabilities of each class, based on the training data
        DistributionParameters  % the parameters of each Normal distribution (means and standard deviations)
    end
    
    methods   
        function obj = my_ClassificationNaiveBayes(X, Y)
           
            % assign superclass properties
            obj@my_ClassificationModel(X,Y);

            % assign class specific properties
            obj.DistributionParameters = {};
            obj.Prior = [];
            
            % for each class in the problem:
            for i = 1:length(obj.ClassNames)
                
                % grab the current class name:
				this_class = obj.ClassNames(i);
                % get all the examples belonging to this class:
                examples_from_this_class = obj.X(obj.Y==this_class,:);
                
                % count them, and divide by the total number of examples to 
                % estimate the prior probability of observing this class:
                obj.Prior(end+1) = size(examples_from_this_class,1) / obj.NumObservations;
                                
                % and estimate the parameters of a Normal distribution from
                % the values seen for each feature (within this class):
                % (for loop over the features for clarity):
                for j = 1:size(obj.X, 2)
                    
                    % mean and standard deviation:
                    obj.DistributionParameters{i,j} = [mean(examples_from_this_class(:,j)); std(examples_from_this_class(:,j))];
                end                 
            end           
        end
        
        function [predictions, scores] = predict(obj, test_examples)           
            % get ready to store our predicted class labels:
            predictions = categorical;
            
            % write something to the last element you'll use:
            predictions(height(test_examples), 1) = obj.ClassNames(1);
            % or call zeros() (if it's a numerical array):
            posterior_ = zeros(1, length(obj.ClassNames));
            scores = zeros(height(test_examples), length(obj.ClassNames));
            
            % for all the testing examples we've been passed:
            for i=1:size(test_examples,1)
                % grab the next testing example:
                this_test_example = test_examples(i,:);

                % for each class, calcuate a value proportional to the
                % posterior probability by multiplying the likelihood by
                % the prior (Bayes theorem):
                for j=1:length(obj.ClassNames)

                    % (we need to multiply lots of individual likelihoods
                    % (per feature value) together; starting off with 1
                    % lets us write a loop that just does multiplications)
                    this_likelihood = 1;
            
                    % get the overall likelihood of the current example by
                    % multiplying together individual likelihoods for each
                    % feature value in it (treating them as independent
                    % events, as per the class conditional independence
                    % assumption):
                    for k=1:length(this_test_example)
                        % individual likelihoods of each feature value,
                        % given this class, come from the Normal
                        % distributions we estimated for this class during
                        % fitting:
                        this_likelihood = this_likelihood * obj.calculate_pd(this_test_example(k), obj.DistributionParameters{j,k}(1), obj.DistributionParameters{j,k}(2));
                    end
                                        
                    % get the prior probability for this class:
                    this_prior = obj.Prior(j);
                    
                    % multiply the likelihood and the prior for a value
                    % proportional (not equal) to the posterior (hence the
                    % underscore):
                    posterior_(j) = this_likelihood * this_prior;
            
                end

                
                % which class had the highest posterior probability (and is
                % therefore the most likely class label):
                [~, winning_index] = max(posterior_);
                % set this as the prediction for this example:
                this_prediction = obj.ClassNames(winning_index);

                % add it to our array of predictions, and move on to the
                % next example (next iteration of this for loop):
                predictions(i,1) = this_prediction;

                % calc scores
                scores(i,:) = posterior_ ./ sum(posterior_);
            end           
        end
    end

    methods (Static)
        % calculate the value of a Normal probability density function
        % (described by mu, sigma) for a feature value x
        function pd = calculate_pd(x, mu, sigma)
			first_bit = 1 / sqrt(2*pi*sigma^2);
            second_bit = - ( ((x-mu)^2) / (2*sigma^2) );
            pd = first_bit * exp(second_bit);
        end 
    end   
end