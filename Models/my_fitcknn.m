% Description: create a model ready to perform k-NN classification from
% some training data 
%
% Inputs:
% train_examples: a numeric array containing the training examples
% train_labels: a categorical array containing the associated
% labels (i.e., with the same ordering as train_examples)
% 
% Optionally, the user can also ask to set 'NumNeighbors' (via an extra
% name-value pair) controlling the number of nearest neighbours used during
% prediction (1 by default)
% 
% Optionally, the user can also ask to switch on 'Verbose' mode (via an
% extra name-value pair) causing the model to generate debug
%
% Outputs:
% m: a my_ClassificationKNN object holding the parameters of the resulting
% model
% 
% Notes:
% The only job here is to instantiate an object from the relevant
% class and hand it back; just providing equivalence with the Matlab
% implementation for familiarity/ease of use
%
function m = my_fitcknn(train_examples, train_labels, varargin)

    % take an extra name-value pair allowing us to turn debug on:
    p = inputParser;
    % take an extra name-value pair allowing us to set the number of
    % nearest neighbours used during prediction (1 by default):
    addParameter(p, 'NumNeighbors', 1);

    p.parse(varargin{:});
    
    % use the supplied parameters to create a new my_ClassificationKNN
    % object:
    m = my_ClassificationKNN(train_examples, train_labels, ...
        p.Results.NumNeighbors);
            
end