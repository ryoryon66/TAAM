import sys
sys.path.append("./model_src")
sys.path.append("./task/")

from TAAMmodel import TAAMModel, BooleanAlgebra, Interpretation, TypedGraph, powerset
from random_test1 import ConstraintChecker
from curses.ascii import isascii
from lib2to3.pgen2.pgen import generate_grammar
from pprint import pprint
from re import T
import time
import random
from typing import Any, Generator, Union, cast, Optional, TypeAlias
from sympy import S
import pandas as pd



SEED = time.time()
random.seed(SEED)


class StopWatch():

    def __init__(self, model: TAAMModel):
        self.model = model
        self.result: dict[str, tuple[Optional[float], Optional[bool]]] = {constraint_name: (
            None, None) for constraint_name in ConstraintChecker.constraints_names}  # (time,result_TF)

        return

    def measure_time(self, constraints_names: list[str] = ConstraintChecker.constraints_names) -> dict[str, tuple[Optional[float], Optional[bool]]]:

        for constraint_name in constraints_names:
            checker = ConstraintChecker(self.model, [], [constraint_name])
            start_time = time.time()
            is_satisfied: bool = checker.check_constraint(constraint_name)
            end_time = time.time()
            self.result[constraint_name] = (
                end_time - start_time, is_satisfied)

        return self.result


class DoExperiment:

    def __init__(self, filepath: str):
        self.filepath = filepath

        return

    def generate_model(self):

        num_node = random.randint(1, 10)
        num_edge: int = int((num_node ** 2) * random.random()
                            * (random.random() ** 0.3))

        num_onode = random.randint(0, num_node)
        num_pnode = num_node - num_onode

        Aord_size = num_onode
        Themes_size = random.randint(1, 4)
        limit_num_given_themes = random.randint(1, 4)

        num_propvar = random.randint(1, 2)

        limit_image_size = random.randint(1, 4)

        graph_params = {
            "Aord_size": Aord_size,
            "Themes_size": Themes_size,
            "num_pnode": num_pnode,
            "num_onode": num_onode,
            "num_edge": num_edge,
            "limit_num_given_themes": limit_num_given_themes
        }

        boolean_algebra_params = {
            "num_propvar": num_propvar
        }

        interpretation_params = {
            "limit_image_size": limit_image_size
        }

        pprint(graph_params)
        pprint(boolean_algebra_params)
        pprint(interpretation_params)

        typed_graph = TypedGraph(**graph_params)
        boolean_algebra = BooleanAlgebra(**boolean_algebra_params)
        interpretation = Interpretation(
            typed_graph, boolean_algebra, **interpretation_params)

        model = TAAMModel(typed_graph, boolean_algebra, interpretation)

        return model

    def generate_time_data(self, iterate_num: int = 100):

        cnt = 1

        while iterate_num > 0:

            print(f"model {cnt}")
            cnt += 1

            iterate_num = iterate_num - 1
            model = self.generate_model()
            result = StopWatch(model).measure_time()
            self.write_time_data(model, result)

        return

    def write_time_data(self, model: TAAMModel, result: dict[str, tuple[Optional[float], Optional[bool]]]):

        # df = pd.read_csv(self.filepath,encoding="utf_8",)

        # model data
        new_row: dict = {
            "Aord_size": len(model.typed_graph.Aord),
            "Themes_size": len(model.typed_graph.Themes),
            "num_nodes": len(list(model.typed_graph.graph.nodes)),
            "num_pnode": len(list(model.typed_graph.enumerate_pnode())),
            "num_onode": len(list(model.typed_graph.enumerate_onode())),
            "num_edge": len(list(model.typed_graph.graph.edges)),
            "max_given_themes_num": -1,

            "num_propvar": model.D.NUM_PROPVAR,

            "max_given_logics_num": -1,

        }

        for node in model.typed_graph.graph.nodes:
            num_themes = len(model.typed_graph.graph.nodes[node]["themes"])
            new_row["max_given_themes_num"] = max(
                num_themes, new_row["max_given_themes_num"])

        for edge in model.typed_graph.graph.edges:
            num_themes = len(model.typed_graph.graph.edges[edge]["themes"])
            new_row["max_given_themes_num"] = max(
                num_themes, new_row["max_given_themes_num"])

        for image in model.I.mapping.values():
            num_logics = len(image)
            new_row["max_given_logics_num"] = max(
                num_logics, new_row["max_given_logics_num"])

        # data about the constraints

        for constraint_name in result.keys():
            new_row[f"{constraint_name}_time"] = result[constraint_name][0]
            new_row[f"{constraint_name}_TF"] = result[constraint_name][1]

        print(new_row)

        # write the first data with this code.
        # import csv
        # labels = list(new_row.keys())
        # dct_arr = [
        #     new_row
        # ]

        # try:
        #     with open(self.filepath, 'a') as f:
        #         writer = csv.DictWriter(f, fieldnames=labels)
        #         writer.writeheader()
        #         for elem in dct_arr:
        #             writer.writerow(elem)
        # except IOError:
        #     print("I/O error")

        df: pd.DataFrame = pd.DataFrame(
            pd.read_csv(self.filepath, encoding="utf_8"))
        df = df.append(new_row, ignore_index=True, sort=False)
        df.to_csv(self.filepath, index=False)

        return


if __name__ == "__main__":

    experiment = DoExperiment("./data/experiment1/time_data.csv")
    experiment.generate_time_data(1000000000000)
