"""The underlying data structure used in gplearn.

The :mod:`gplearn._program` module contains the underlying representation of a
computer program. It is used for creating and evolving programs used in the
:mod:`gplearn.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

from copy import copy

import numpy as np
import random
from sklearn.utils.random import sample_without_replacement

from .functions import _Function
from .utils import check_random_state
from .config import functions_arity, window_lengths
import warnings
import qlib
from qlib.data import D
from backtest.ictester import ICBacktester

def depth(program: list):
    """Calculates the maximum depth of the program tree."""
    terminals = [0]
    depth = 1
    for node in program:
        if isinstance(node, str) and node in functions_arity:
            if functions_arity[node] == 4:
                terminals.append(2)
            else:
                terminals.append(functions_arity[node])
            depth = max(len(terminals), depth)
        else:
            terminals[-1] -= 1
            while terminals[-1] == 0:
                terminals.pop()
                terminals[-1] -= 1
    return depth - 1


class _Program(object):

    """A program-like representation of the evolved program.

    This is the underlying data-structure used by the public classes in the
    :mod:`gplearn.genetic` module. It should not be used directly by the user.

    Parameters
    ----------
    function_set : list
        A list of valid functions to use in the program.

    arities : dict
        A dictionary of the form `{arity: [functions]}`. The arity is the
        number of arguments that the function takes, the functions must match
        those in the `function_set` parameter.

    init_depth : tuple of two ints
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    n_features : int
        The number of features in `X`.

    const_range : tuple of two floats
        The range of constants to include in the formulas.

    metric : _Fitness object
        The raw fitness metric.

    p_point_replace : float
        The probability that any given node will be mutated during point
        mutation.

    parsimony_coefficient : float
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

    random_state : RandomState instance
        The random number generator. Note that ints, or None are not allowed.
        The reason for this being passed is that during parallel evolution the
        same program object may be accessed by multiple parallel processes.

    transformer : _Function object, optional (default=None)
        The function to transform the output of the program to probabilities,
        only used for the SymbolicClassifier.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, a new naive
        random tree will be grown. If provided, it will be validated.

    Attributes
    ----------
    program : list
        The flattened tree representation of the program.

    raw_fitness_ : float
        The raw fitness of the individual program.

    fitness_ : float
        The penalized fitness of the individual program.

    oob_fitness_ : float
        The out-of-bag raw fitness of the individual program for the held-out
        samples. Only present when sub-sampling was used in the estimator by
        specifying `max_samples` < 1.0.

    parents : dict, or None
        If None, this is a naive random program from the initial population.
        Otherwise it includes meta-data about the program's parent(s) as well
        as the genetic operations performed to yield the current program. This
        is set outside this class by the controlling evolution loops.

    depth_ : int
        The maximum depth of the program tree.

    length_ : int
        The number of functions and terminals in the program.

    """

    def __init__(self,
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 transformer=None,
                 feature_names=None,
                 program=None,
                 qlib_config = {
                     "data_client": D, 
                     "instruments": D.instruments(market="all"), 
                     "start_time": '2020-01-01',  
                     "end_time": '2020-12-31', 
                     "freq": "day", 
                 }):

        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.transformer = transformer
        self.feature_names = feature_names
        self.program = program
        self.qlib_config = qlib_config
        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        else:
            # Create a naive random program
            self.program = self.build_program(random_state)

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None

    def build_program(self, random_state):
        """Build a naive random program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)

        # Start a program with a function to avoid degenerative programs
        function = random_state.randint(len(self.function_set))
        function = self.function_set[function]
        program = [function]
        terminal_stack = [functions_arity[function]]

        while terminal_stack:
            depth = len(terminal_stack)
            choice = self.n_features + len(self.function_set)
            choice = random_state.randint(choice)
            # Determine if we are adding a function or terminal
            if (depth < max_depth) and (method == 'full' or
                                        choice <= len(self.function_set)):
                function = random_state.randint(len(self.function_set))
                function = self.function_set[function]
                program.append(function)
                terminal_stack.append(functions_arity[function])
            else:
                # We need a terminal, add a variable or constant
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                    program.append(terminal)
                    
                else:
                    if self.feature_names:
                        program.append(self.feature_names[terminal])
                    else:
                        program.append(terminal)
                        warnings.warn("No feature names passed to the program.", RuntimeWarning)
                
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0 or terminal_stack[-1] == 3:
                    if terminal_stack[-1] == 0:
                        terminal_stack.pop()
                    else:
                        terminal = random_state.randint(len(window_lengths))
                        terminal = window_lengths[terminal]
                        program.append(terminal)
                        terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1

        # We should never get here
        return None

    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        terminals = [0]
        for node in self.program:
            if isinstance(node, str) and node in functions_arity:
                terminals.append(functions_arity[node])
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0 or  terminals[-1] == 2:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        apply_stack = []
        expression = ""
        if len(self.program) == 1:
            return str(self.program[0])
        
        for node in self.program:

            if isinstance(node, str):
                if node in functions_arity:
                    arity = 2 if functions_arity[node] == 4 else functions_arity[node]
                    apply_stack.append(arity)
                    expression = expression + node + "("
                elif node in self.feature_names:
                    expression = expression + node
                    apply_stack[-1] -= 1
                    if apply_stack[-1] == 1:
                        expression = expression + ", "
                    else:
                        while apply_stack[-1] == 0:
                            expression = expression + ")"
                            apply_stack.pop()
                            if not apply_stack:
                                break
                            apply_stack[-1] -= 1
                            if apply_stack[-1] == 1:
                                expression = expression + ", "
                else:
                    raise TypeError(f"Unknown Function or Feature {node}")
            
            else:
                expression = expression + str(node)
                apply_stack[-1] -= 1
                if apply_stack[-1] == 1:
                    expression = expression + ", "
                else:
                    while apply_stack[-1] == 0:
                        expression = expression + ")"
                        apply_stack.pop()
                        if not apply_stack:
                            break
                        apply_stack[-1] -= 1
                        if apply_stack[-1] == 1:
                            expression = expression + ", "
        return expression

    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.

        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.

        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.

        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, str) and node in functions_arity:
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([functions_arity[node], i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                if isinstance(node, int):
                    if self.feature_names is None:
                        feature_name = 'X%s' % node
                    else:
                        feature_name = self.feature_names[node]
                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                else:
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, str) and node in functions_arity:
                if functions_arity[node] == 4:
                    terminals.append(2)
                else:
                    terminals.append(functions_arity[node])
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)

    def execute(self, X_shape):
        """Execute the program.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        # Check for single-node programs
        node = self.program[0]
        if isinstance(node, float):
            return np.repeat(node, X_shape[0])

        if isinstance(node, int):
            return np.repeat(node, X_shape[0])

        apply_stack = []
        expression = ""

        for node in self.program:

            if isinstance(node, str):
                if node in functions_arity:
                    arity = 2 if functions_arity[node] == 4 else functions_arity[node]
                    apply_stack.append(arity)
                    expression = expression + node + "("
                elif node in self.feature_names:
                    expression = expression + node
                    apply_stack[-1] -= 1
                    if apply_stack[-1] == 1:
                        expression = expression + ", "
                    else:
                        while apply_stack[-1] == 0:
                            expression = expression + ")"
                            apply_stack.pop()
                            if not apply_stack:
                                break
                            apply_stack[-1] -= 1
                            if apply_stack[-1] == 1:
                                expression = expression + ", "
                else:
                    raise TypeError(f"Unknown Function or Feature {node}")
            
            else:
                expression = expression + str(node)
                apply_stack[-1] -= 1
                if apply_stack[-1] == 1:
                    expression = expression + ", "
                else:
                    while apply_stack[-1] == 0:
                        expression = expression + ")"
                        apply_stack.pop()
                        if not apply_stack:
                            break
                        apply_stack[-1] -= 1
                        if apply_stack[-1] == 1:
                            expression = expression + ", "                

        print("executing: ", expression)
        try:
            data = self.qlib_config["data_client"].features(
                self.qlib_config["instruments"],
                [expression],
                start_time=self.qlib_config["start_time"],
                end_time=self.qlib_config["end_time"],
                freq=self.qlib_config["freq"]
            )
        except Exception:
            print(f"{expression} can not be executed.")
            data = self.qlib_config["data_client"].features(
                self.qlib_config["instruments"],
                ["$close"],
                start_time=self.qlib_config["start_time"],
                end_time=self.qlib_config["end_time"],
                freq=self.qlib_config["freq"]
            )
        # We should never get here
        return data.squeeze().to_numpy()

    def get_all_indices(self, n_samples=None, max_samples=None,
                        random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]

    def raw_fitness(self, X_shape, y, sample_weight):
        """Evaluate the raw fitness of the program according to program and y.

        Parameters
        ----------
        X_shape: tuple-like, shape of X

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        y_pred = self.execute(X_shape)
        if len(y) != len(y_pred):
            raise ValueError("Different length between y and y_pred.")
        
        else:
            print("##", str(self.__str__()), "##")
            raw_fitness = ICBacktester(self.__str__(), self.qlib_config["start_time"], self.qlib_config["end_time"], self.qlib_config["instruments"], self.qlib_config["freq"]).calculate1()
            print("ic: ", raw_fitness, type(raw_fitness))
        return abs(raw_fitness)

    def fitness(self, parsimony_coefficient=None):
        """Evaluate the penalized fitness of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        penalty = parsimony_coefficient * len(self.program) * self.metric.sign
        return self.raw_fitness_ - penalty

    def get_subtree(self, random_state, program=None, target_depth=None):
        """Get a subtree from the program, optionally constrained to a specific depth.

        Parameters
        ----------
        random_state : RandomState instance

        program : list, optional (default=None)

        target_depth : int, optional (default=None)

        Returns
        -------
        start, end : tuple of two ints
        """
        if program is None:
            program = self.program

        depths = []
        stack = []
        for node in program:
            depths.append(len(stack))
            if isinstance(node, str) and node in functions_arity:
                if functions_arity[node] == 4:
                    stack.append(2)
                else:
                    stack.append(functions_arity[node])
            else:
                if stack:
                    stack[-1] -= 1
                    while stack and stack[-1] == 0:
                        stack.pop()
                        if not stack:
                            break
                        stack[-1] == 0

        if target_depth is not None:
            candidates = [i for i, d in enumerate(depths) if d == target_depth]
            if len(candidates) > 0:
                start = random_state.choice(candidates)
            else:
                raise ValueError("The specified depth has no nodes.")
        else:
            probs = np.array([0.9 if isinstance(n, str) and n in functions_arity else 0.1
                              for n in program])
            probs = np.cumsum(probs / probs.sum())
            start = np.searchsorted(probs, random_state.uniform())

        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, str) and node in functions_arity:
                if functions_arity[node] == 4:
                    stack += 2
                else:
                    stack += functions_arity[node]
            end += 1

        return start, end

    def reproduce(self):
        """Return a copy of the embedded program."""
        return copy(self.program)

    def crossover(self, donor, random_state):  # donor is a program list
        """Perform the crossover genetic operation on the program.

        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.

        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """

        if depth(donor) != self._depth():
            raise ValueError("Crossover two trees with different depth.")
        subtree_depth = random.randint(1, self._depth() - 1)
        # Get a subtree to replace
        start, end = self.get_subtree(random_state = random_state, target_depth = subtree_depth)
        # Get a subtree to donate
        donor_start, donor_end = self.get_subtree(random_state = random_state, program = donor, target_depth = subtree_depth)
        # Insert genetic material from donor
        # return (self.program[:start] +
        #         donor[donor_start:donor_end] +
        #         self.program[end:]), removed, donor_removed

        return (self.program[:start] + donor[donor_start:donor_end] + self.program[end:]), (donor[:donor_start] + self.program[start:end] + donor[donor_end:])
    
    def depth_evolution(self, partner, random_state):  # partner is a _Program
        function = random_state.randint(len(self.function_set))
        function = self.function_set[function]
        while functions_arity[function] == 4:
            function = random_state.randint(len(self.function_set))
            function = self.function_set[function]            
        if functions_arity[function] == 1:
            return [function] + self.program            
        return [function] + self.program + partner.program

    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)
