import sys
sys.path.append("./model_src")
sys.path.append("./task/")

from TAAMmodel import TAAMModel, BooleanAlgebra, Interpretation, TypedGraph, powerset
from random_test1 import ConstraintChecker
from lib2to3.pgen2.pgen import generate_grammar
from re import T
import time
import random
from sympy import S



SEED = time.time()
random.seed(SEED)


class TrivialInterpretation(Interpretation):

    def __init__(self, typed_graph: TypedGraph, D: BooleanAlgebra):

        # initialize all interpretations to empty sets.
        super().__init__(typed_graph, D, 6)

        node = list(typed_graph.graph.nodes)[0]
        attached_t = typed_graph.graph.nodes[node]["themes"][0]
        propvar = D.PROPVARS[0]
        assert (node == "0")
        print(attached_t)

        self.mapping[((attached_t), node)] = [propvar]

        for T in powerset(typed_graph.Themes):

            assert (list(T) == sorted(list(T)))

            if attached_t in T:
                self.mapping[(T, node)] = [propvar]
            else:
                self.mapping[(T, node)] = []

        self.mapping[((attached_t), self.OMEGA)] = [
            propvar, ~propvar, S.true, S.false]

        for T in powerset(typed_graph.Themes):

            assert (list(T) == sorted(list(T)))

            if attached_t in T:
                self.mapping[(T, self.OMEGA)] = [
                    propvar, ~propvar, S.true, S.false]
            else:
                self.mapping[(T, self.OMEGA)] = []

        return


def generate_trivial_model(themes_size: int = 6):

    # one node and no edges
    g = TypedGraph(Aord_size=1, Themes_size=themes_size, num_pnode=0,
                   num_onode=1, num_edge=0, limit_num_given_themes=1)

    # only one propositional variable
    d = BooleanAlgebra(1)

    #
    I = TrivialInterpretation(g, d)

    # print(I)

    model = TAAMModel(g, d, I)

    return model


if __name__ == "__main__":

    model = generate_trivial_model()

    antecedent = ["aass", "i", "vi", "bat", "pr", "mat", "manss", "ss"]
    consequent = ["tr", "nnp", "nsa", "kos", "nss"]

    checker = ConstraintChecker(model, antecedent, consequent)
    result, counter = checker.run_checker()

    print(result, counter)
    model.visualize(
        description=f"{model.typed_graph}\nantecedent:{antecedent}\nconsequent:{consequent}\nsatisfied_constraints_num :{counter}\n\nI:{model.I}",
        title=f"trivial_model{result}")

    assert (result)
