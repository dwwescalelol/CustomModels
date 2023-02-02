classdef kd_Node
    %kd_Node Node for a kd_tree, different from regular b-node as stores index.
    % D-Tree is a binary tree, thus left and right are only children.
    % KDNode Properties:
    %    Obs    - Single training observation.
    %    Index          - Index of training observation in training data.
    %    Left           - Left child.
    %    Right          - Right child.
    %    Limits         - Bounds that Obs exists in.
    properties
        Obs
        Index
        Left
        Right
        Limits
    end
    
    methods
        function obj = kd_Node(obs, index, limits)
            %kd_Node Construct an instance of this class.
            obj.Obs = obs;
            obj.Index = index;
            obj.Limits = limits;
        end
    end

end