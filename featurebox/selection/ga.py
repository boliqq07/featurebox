import random
from collections import deque
from functools import partial

import numpy as np
from deap import base
from deap import tools
from deap.algorithms import varAnd
from deap.tools import mutShuffleIndexes
from mgetool.newclass import create
from mgetool.tool import batch_parallelize, check_random_state
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectorMixin
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.utils.validation import check_is_fitted

from featurebox.selection.mutibase import MutiBase


def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None, n_jobs=2,
             halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    # n_job=n
    invalid_ind2 = [tuple(i) for i in invalid_ind]
    fitnesses = batch_parallelize(n_jobs, toolbox.evaluate, invalid_ind2, batch_size=30, tq=True)
    # n_job=1
    # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    best = deque([halloffame.items[0].fitness.values[0]], maxlen=15)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Append the current generation statistics to the logbook
        record = stats.compile(population + halloffame.items) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - halloffame.maxsize)
        offspring.extend(halloffame.items)
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        offspring = toolbox.map(toolbox.filt, offspring)
        offspring = list(offspring)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # n_job=n
        invalid_ind2 = [tuple(i) for i in invalid_ind]

        fitnesses = batch_parallelize(n_jobs, toolbox.evaluate, invalid_ind2, batch_size=30, tq=True)
        # fitnesses = parallelize(n_jobs, toolbox.evaluate, invalid_ind2)
        # n_job=1
        # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        best.append(halloffame.items[0].fitness.values[0])
        if sum(best) / best.maxlen == best[-1]:
            break
        # Replace the current population by the offspring
        population[:] = offspring

    return population, logbook


def generate_xi():
    return random.randint(0, 1)


def generate(space):
    return [generate_xi() for _ in range(space)]


def filt(ind, min_=2, max_=None):
    if max_ is not None:
        if np.sum(ind) > max_:
            index = np.where(np.array(ind) == 1)[0]
            k = random.randint(min_, max_)
            index2 = random.sample(list(index), k=k)
            ind[:] = [0] * len(ind)
            [ind.__setitem__(i, 1) for i in index2]

        elif np.sum(ind) < min_:
            k = random.randint(min_, max_)
            index2 = random.sample(list(range(len(ind))), k=k)
            ind[:] = [0] * len(ind)
            [ind.__setitem__(i, 1) for i in index2]

    else:
        if np.sum(ind) < min_:
            k = random.randint(min_, len(ind))
            index2 = random.sample(list(range(len(ind))), k=k)
            ind[:] = [0] * len(ind)
            [ind.__setitem__(i, 1) for i in index2]

    # if np.sum(ind)<min_:
    #     raise UserWarning("???")

    return ind


class GA(BaseEstimator, MetaEstimatorMixin, SelectorMixin, MutiBase):
    """
    GA with binding. Please just passing training data.

    Examples
    ---------
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.svm import SVR
    >>> data = load_boston()
    >>> x = data.data[:50]
    >>> y = data.target[:50]
    >>> svr = SVR(gamma="scale", C=100)
    >>> ga = GA(estimator=svr, n_jobs=2, pop_n=50, hof_n=1, cxpb=0.8, mutpb=0.4, ngen=3, max_or_min="max", mut_indpb=0.1, min_=2, muti_index=[0, 5],random_state=0)
    >>> ga.fit(x_test, y_test)

    Then
    ::

        ga.score_cv(x, y)

    """

    def __init__(self, estimator, n_jobs=2, pop_n=1000, hof_n=1, cxpb=0.6, mutpb=0.3, ngen=40, max_or_min="max",
                 mut_indpb=0.05, max_=None, min_=2, random_state=None, muti_grade=2, muti_index=None):
        """

        Parameters
        ----------
        estimator:
            sklearn estimator
        n_jobs:int
            njobs
        pop_n:int
            population
        hof_n:int
            hof
        cxpb:float
            probility of cross
        mutpb:float
            probility of mutate
        ngen:int
            generation
        max_or_min:str
            "max","min";max problem or min
        mut_indpb:float
            probility of mutate of each node.
        max_:int
            max size
        min_:int
            min size
        random_state:float
            randomstate
        muti_grade:
            binding grade
        muti_index:
            binding range [min,max]
        """
        super().__init__(muti_grade=muti_grade, muti_index=muti_index, must_index=None)
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.pop_n = pop_n
        self.hof_n = hof_n
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen
        self.mut_indpb = mut_indpb
        self.max_ = max_
        self.min_ = min_
        self.max_or_min = max_or_min
        self.random_state = random_state

        check_random_state(random_state)
        random.seed(random_state)
        np.random.seed(0)

        self.toolbox = base.Toolbox()
        if max_or_min == "max":
            FitnessMax = create("FitnessMax", base.Fitness, weights=(1.0,))
        else:
            FitnessMax = create("FitnessMax", base.Fitness, weights=(-1.0,))
        self.Individual = create("Individual", list, fitness=FitnessMax)
        self.toolbox = base.Toolbox()
        # Attribute generator

    def generate_min_max(self, space, min_=2, max_=None):
        ind = [generate_xi() for _ in range(space)]
        return filt(ind, min_=min_, max_=max_)

    def feature_fold_length(self, feature):

        muti_grade, muti_index = self.muti_grade, self.muti_index
        if self.check_muti:
            cc = []
            feature = np.sort(feature)

            i = 0
            while i <= feature[-1]:
                if muti_index[0] <= i < muti_index[1]:
                    i += self.muti_grade
                    cc.append(self.muti_grade)
                else:
                    i += 1
                    cc.append(1)
            return cc
        else:
            return [1] * len(feature)

    def fit(self, X, y):
        """Fit data and run GA."""
        x_space = X.shape[1]
        self.X = X
        self.y = y
        self.x_space_fold = self.feature_fold_length(list(range(x_space)))
        toolbox = self.toolbox
        toolbox.register("generate_x", self.generate_min_max, len(self.x_space_fold), min_=self.min_, max_=self.max_)
        # Structure initializers
        toolbox.register("individual", tools.initIterate, self.Individual,
                         toolbox.generate_x)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        fit_func = partial(self.fitness_func, model=self.estimator, x=X, y=y, return_model=False)
        toolbox.register("evaluate", fit_func)
        toolbox.register("filt", filt, min_=self.min_, max_=self.max_)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", mutShuffleIndexes, indpb=self.mut_indpb)
        toolbox.register("select", tools.selTournament, tournsize=3)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop = toolbox.population(n=self.pop_n)
        self.hof = tools.HallOfFame(self.hof_n)

        eaSimple(pop, self.toolbox, cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.ngen, n_jobs=self.n_jobs,
                 stats=stats, halloffame=self.hof, verbose=True)
        for i in self.hof.items:
            print(self.unfold(i), i.fitness)
        support_ = self.unfold(self.hof.items[0])
        self.support_ = np.array(support_) > 0
        return self

    def unfold(self, ind):
        sss = []
        [sss.extend([i] * j) for i, j in zip(ind, self.x_space_fold)]
        return sss

    def fitness_func(self, ind, model, x, y, return_model=False, cv=5):
        sss = self.unfold(ind)
        index = np.where(np.array(sss) == 1)[0]
        x = x[:, index]
        if x.shape[1] > 1:
            svr = model
            svr.fit(x, y)
            if hasattr(svr, "best_score_"):
                sc = svr.best_score_
            else:
                sc = np.mean(cross_val_score(svr, x, y, cv=cv))
            if return_model:
                return sc, svr
            else:
                return sc,
        else:
            if return_model:
                return 0, None
            else:
                return 0,

    def socre_func(self, ind, model, x, y):
        sss = self.unfold(ind)
        index = np.where(np.array(sss) == 1)[0]
        x = x[:, index]
        if x.shape[1] > 1:
            svr = model
            y2 = svr.predict(x)
            sc = r2_score(y, y2)
            return sc
        else:
            raise TypeError("only one feature,error")

    def predict_func(self, ind, model, x):
        sss = self.unfold(ind)
        index = np.where(np.array(sss) == 1)[0]
        x = x[:, index]
        if x.shape[1] > 1:
            svr = model
            y = svr.predict(x)
            return y
        else:
            raise TypeError("only one feature,error")

    def _get_support_mask(self):
        check_is_fitted(self, 'support_')
        return self.support_

    def score_cv(self, X, y):
        """Reduce X to the selected feature and then return the score of the underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_feature]
            The input0 samples.

        y : array of shape [n_samples]
            The target values.
        """

        mod = self.fitness_func(self.hof.items[0], self.estimator, self.X, self.y, return_model=True)[1]
        score = self.fitness_func(self.hof.items[0], mod, X, y, return_model=False)
        return score

    def score(self, X, y):
        """Reduce X to the selected feature and then return the score of the underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_feature]
            The input0 samples.

        y : array of shape [n_samples]
            The target values.
        """

        mod = self.fitness_func(self.hof.items[0], self.estimator, self.X, self.y, return_model=True)[1]
        score = self.socre_func(self.hof.items[0], mod, X, y)
        return score

    def predict(self, X):
        """Reduce X to the selected feature and then return the score of the underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_feature]
            The input0 samples.

        """

        mod = self.fitness_func(self.hof.items[0], self.estimator, self.X, self.y, return_model=True)[1]
        score = self.predict_func(self.hof.items[0], mod, X)
        return score


if __name__ == "__main__":
    data = load_boston()
    x = data.data
    y = data.target
    svr = SVR(gamma="scale", C=100)

    ga = GA(estimator=svr, n_jobs=1, pop_n=100, hof_n=1, cxpb=0.8, mutpb=0.4, ngen=10,
            max_or_min="max", mut_indpb=0.1, min_=2, muti_index=[0, 5], random_state=0)
    ga.fit(x, y)
    ga.score(x, y)
