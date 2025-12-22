function [bestValue, bestCap] = runTraditionalOptimisation(lb, ub, w_obj, envOpts)
% runTraditionalOptimisation  Optimise the capacity sizing problem using a
% particle swarm algorithm.  This helper function wraps the objective
% evaluation such that the optimiser only needs to supply a vector of
% decision variables (capacities).  The weight vector w_obj and
% environment options envOpts are captured in the nested objective
% function.

% Number of optimisation variables
nVars = numel(lb);

% Define objective function as a closure capturing w_obj and envOpts.  The
% psoObjectiveWrapper resets the environment, runs a single evaluation and
% returns the negative of the score (since particleswarm performs
% minimisation).  The environment reset and cost computation are
% contained within the wrapper.

    function cost = psoObjectiveWrapper(x)
        % Evaluate the candidate vector x.  The action vector x
        % contains both capacity and location parameters.  We reset
        % the environment to its initial state and then perform a
        % single evaluation of the objective using capStepWrapper.
        try
            % Ensure x is a column vector
            x = x(:);
            % Reset the environment to its initial state.  The reset
            % function ignores the current candidate values – the
            % environment state is internal to the simulation and is
            % controlled entirely by capStepWrapper.
            capResetWrapper(envOpts.cap_init, envOpts);
            % Obtain the reward for this candidate design.  The
            % capStepWrapper function handles the full simulation
            % (including iter_couple_most_mpng_24h_merged) and
            % computes the multiobjective score according to w_obj.
            reward = capStepWrapper(x, w_obj);
            % Convert reward to a cost for minimisation
            cost = -double(reward);
        catch evalErr
            % In case of error, penalise heavily so optimiser avoids
            % invalid regions
            warning('PSO objective encountered error: %s', evalErr.message);
            cost = 1e6;
        end
    end
% Configure particle swarm options
psOpts = optimoptions('particleswarm', ...
    'SwarmSize', 30, ...
    'MaxIterations', 50, ...
    'Display', 'off');

% Run particle swarm optimisation
[bestCap, bestVal] = particleswarm(@psoObjectiveWrapper, nVars, lb, ub, psOpts);
bestValue = bestVal;

end

