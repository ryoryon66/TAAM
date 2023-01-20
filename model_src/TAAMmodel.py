from __future__ import annotations


import pprint
import random
from typing import Any, Generator, Union, cast, Optional, TypeAlias
from matplotlib import colors
import networkx as nx
import matplotlib.pyplot as plt
import time
import itertools
import sys
import dill
import traceback
import sympy
from sympy import S, Equivalent, simplify_logic

# Proposition
Prop: TypeAlias = sympy.core.symbol.Symbol

# Logical Expression
Expr: TypeAlias = Union[
    Prop,
    sympy.logic.boolalg.BooleanTrue,
    sympy.logic.boolalg.BooleanFalse,
    sympy.logic.boolalg.BooleanFunction
]

sys.setrecursionlimit(10 ** 8)


SEED = int(time.time() * 10)


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


class TypedGraph():

    def __init__(self, Aord_size=6, Themes_size=10, num_pnode=5, num_onode=3, num_edge=10, limit_num_given_themes=3) -> None:
        """init TypedGraph

        Args:
            Aord_size (int, optional): the number of elements in Aord_size(Aord = {"0","1",... }). Defaults to 6.
            Themes_size (int, optional):the number of Themes (Themes = {"t0","t1",...}). Defaults to 10.
            num_pnode (int, optional): the number of pnodes. Defaults to 5.
            num_onode (int, optional): the number of onodes. Defaults to 3.
            num_edge (int, optional): the number of edges. Defaults to 10.
            limit_num_given_themes (int, optional): Maximum number of themes given to vertices and edges. Defaults to 3.
        """

        assert (num_onode <= Aord_size)
        assert (num_edge <= (num_onode + num_pnode) ** 2)
        assert (limit_num_given_themes > 0)

        self.Aord: list[str] = [str(i) for i in range(Aord_size)]

        # theme is represented in the form of t.number
        self.Themes: list[str] = ["t" + str(i) for i in range(Themes_size)]
        # relationship between nodes.
        self.rel: list[str] = ["attack", "support"]

        self.Apnt = [f"{t}.{a}" for a in self.Aord for t in self.Themes]  \
            + [f"{t}.c" for t in self.Themes]  # use c to represent special symbol

        self.graph = nx.DiGraph()

        # create nodes and attach type attr to nodes

        self.graph.add_nodes_from(random.sample(self.Aord, num_onode))

        possible_pnode = [pnode for pnode in self.Apnt if pnode[pnode.find(
            ".")+1:] in list(self.graph.nodes) + ["c"]]
        self.graph.add_nodes_from(random.sample(
            possible_pnode, min(num_pnode, len(possible_pnode))))

        for node in self.graph.nodes:

            themes_given = random.sample(self.Themes, random.randint(
                1, min(len(self.Themes), limit_num_given_themes)))
            self.graph.nodes[node]["typed"] = sorted(themes_given)
            self.graph.nodes[node]["themes"] = sorted(themes_given)
            del themes_given

        # create edges and attach type attr to edges
        possible_edges = [(u, v)
                          for u in self.graph.nodes for v in self.graph.nodes]
        edges_given = random.sample(
            possible_edges, min(len(possible_edges), num_edge))

        for u, v in edges_given:
            self.graph.add_edge(u, v)
            themes_given = random.sample(self.Themes, random.randint(
                1, min(len(self.Themes), limit_num_given_themes)))
            rel_given = random.sample(
                self.rel, random.randint(1, len(self.rel)))
            types_given = sorted(themes_given) + rel_given
            self.graph.edges[u, v]["typed"] = types_given
            self.graph.edges[u, v]["themes"] = sorted(themes_given)
            self.graph.edges[u, v]["rel"] = rel_given

            if "attack" in rel_given and "support" in rel_given:
                self.graph.edges[u, v]["color"] = "purple"
            elif "attack" in rel_given:
                self.graph.edges[u, v]["color"] = "red"
            else:
                self.graph.edges[u, v]["color"] = "blue"

        assert (self.is_well_formed())

        return

    def is_well_formed(self) -> bool:
        """
        check if this typed graph meet the condition of well-formedness
        """
        graph = self.graph

        for s in graph.nodes:

            if set([]) == set(graph.nodes[s]["typed"]) or set(graph.nodes[s]["typed"]) > set(self.Themes):
                return False

        for e in graph.edges:
            if (set(graph.edges[e]["typed"]) & set(self.Themes)) == set([]):
                return False

        for e in graph.edges:
            if (set(graph.edges[e]["typed"]) & set(self.rel)) == set([]):
                return False

        return True

    def visualize(
        self,
        title="result",
        path_to_save_dir="./",
        node_vis_features: list[str] = ['typed'],
        edge_vis_features: list[str] = ['themes'],
    ):
        """
        Display networkx directed graphs (MultiDiGraph).
        It supports the display of attributes.
        Args:
            notes (str, optional): [description]. Defaults to "".
            title (str, optional): [title to save]. Defaults to "result".
            path_to_save_dir (str, optional): [destination path]. Defaults to "generated_pic/".
            vis_features(list[str],optional): Select attributes to display
        """
        graph: nx.DiGraph = self.graph

        num_vertex = len(list(graph.nodes))

        def attach_scc_id() -> tuple[nx.DiGraph, int]:
            """attach scc_id to nx.Digraph and nx.MultiDigraph

            Attach scc_id (same value for same scc) as an attribute to the graph.
            Returns:
                tuple[nx.DiGraph,int]: Graph after scc_id is assigned, Number of SCCs.
            """
            assert (isinstance(graph, nx.MultiDiGraph)
                    or isinstance(graph, nx.DiGraph))

            scc_id = 1

            for comp in sorted(nx.strongly_connected_components(graph), key=len, reverse=True):
                for node in comp:
                    graph.nodes[node]["scc_id"] = scc_id
                scc_id += 1

            return graph, scc_id-1

        # SCC
        graph, num_scc_group = attach_scc_id()

        def set_node_visinfo(node: int, attributes: list[str] = node_vis_features):
            """
            A function that attaches info to each vertex for use in visualization. 
            Args:
                node (int): a natural number representing a vertex.
                attributes (list[str], optional): key of information to visualize. Defaults to ['skew_type','label','conditions','predicted_labels','scc_id'].
            """

            info = ""

            info = f"node_index:{node}\n"

            for attr in attributes:

                try:

                    info += f"{attr}:{graph.nodes[node][attr]}\n"

                except:
                    pass

            graph.nodes[node]['vis_info'] = info
            graph.nodes[node]["fontsize"] = 15

            return

        def set_edge_visinfo(e: tuple[str, str], attributes: list[str] = edge_vis_features):
            """
            A function that attaches info to each vertex for use in visualization. 
            Args:
                node (int): a natural number representing a vertex.
                attributes (list[str], optional): key of information to visualize. Defaults to ['skew_type','label','conditions','predicted_labels','scc_id'].
            """

            info = ""

            for attr in attributes:

                try:

                    info += f"{attr}:{graph.edges[e][attr]}\n"

                except:
                    pass

            graph.edges[e]['vis_info'] = info
            graph.edges[e]["fontsize"] = 7

            return

        for node in list(graph.nodes):
            set_node_visinfo(node=node)

        for e in graph.edges():
            set_edge_visinfo(e)

        cm_name = 'jet'
        cm = plt.get_cmap(cm_name, num_scc_group+1)

        for node in list(graph.nodes):
            # 0-1 rgb
            color_rgb = cm(graph.nodes[node]['scc_id'])[:3]
            # hexadecimal rgb
            graph.nodes[node]['color'] = colors.to_hex(color_rgb)
            # print(f"type:{type(graph.nodes[node]['color'])},val:{graph.nodes[node]['color']}")
            graph.nodes[node]['penwidth'] = 3

        graph.graph['overlap'] = "prism10000"

        # fdp,sfdp param
        graph.graph['K'] = 1.9
        # sfdp param
        graph.graph['repulsiveforce'] = 1.4

        # convert this into agraphクラス（PyGraphviz）
        G_pgv = nx.nx_agraph.to_agraph(graph)

        for node in G_pgv.nodes():
            G_pgv.get_node(
                node).attr["label"] = graph.nodes[(node)]["vis_info"]

        for u, v in G_pgv.edges():
            G_pgv.get_edge(
                u, v).attr['xlabel'] = f'<<table border="0" cellborder="0"><tr><td bgcolor="gray">{graph.edges[u,v]["vis_info"]}</td></tr></table>>'

        # either fdp or sfdp is recommended.
        G_pgv.draw(path_to_save_dir+f"{title}.pdf", prog='fdp',
                   args='-Gnodesep=1 -Gsize=100 -Gdpi=1000 -Gratio=0.6')

        return

    def enumerate_onode(self) -> Generator[str, None, None]:
        """
        a generator, which enumerates ONODE of the graph

        Yields:
            str: node in ONODE of the graph
        """
        for node in self.graph.nodes:
            node = cast(str, node)
            if node[0] != "t":
                yield node

    def enumerate_pnode(self, form: Optional[str] = None) -> Generator[str, None, None]:
        """
        a generator, which enumerates PNODE of the graph and tells if pnode contains special symbol "c".

        Args:
            form(Optional[str]): this param specifies tha kind of pnode enumerated. choose from ["t.a","t.c",None].if None is specified, vertices that follow one of the forms t.a and t.c are enumerated.

        Yields:
            str: node in PNODE of the graph. 
        """

        if form is None:
            for node in self.graph.nodes:
                node = cast(str, node)
                if node[0] != "t":
                    continue
                yield node
        elif form == "t.a":
            for node in self.graph.nodes:
                node = cast(str, node)
                if node[0] != "t":
                    continue
                if node[-1] != "c":
                    yield node
        elif form == "t.c":
            for node in self.graph.nodes:
                node = cast(str, node)
                if node[0] != "t":
                    continue
                if node[-1] == "c":
                    yield node
        else:
            raise RuntimeError('form should be "t.a", "t.c" or None')

    def __repr__(self) -> str:
        s: str = str(self.graph) + "\n"
        s += f"Aord:{self.Aord}\n"
        s += f"Themes:{self.Themes}\n"
        return s


class BooleanAlgebra():

    def __init__(self, num_propvar=3):
        """
        Boolean Algebra.

        Args:
            num_propvar (int, optional): the number of propositional variables in the boolean Algebra. Defaults to 3.

        """

        self.NUM_PROPVAR = num_propvar

        def gen_props() -> list[Prop]:

            propvar: Union[str, list] = [
                "A" + str(i) for i in range(num_propvar)]
            propvar = [c + "," for c in propvar]
            propvar = cast(str, "".join(propvar))
            propvar = sympy.symbols(propvar)
            propvar = list(propvar)
            return propvar

        self.PROPVARS = gen_props()

        return

    @staticmethod
    def is_equivalent(logic1: Expr, logic2: Expr) -> bool:
        return simplify_logic(Equivalent(logic1, logic2)) == S.true

    @staticmethod
    def is_tautology(logic: Expr) -> bool:
        return BooleanAlgebra.is_equivalent(logic, S.true)

    @staticmethod
    def in_uparrow(logic_min: Expr, logic_tar: Expr) -> bool:
        """
        Determine if uparrow({logic_min}) contains logic_tar

        Args:
            logic_min (Expr): the arg of uparrow
            logic_tar (Expr): the logical expression you want to know if contained or not.
        Returns:
            bool: True if contained.
        """

        assert (not isinstance(logic_min, list))
        assert (logic_min is not None)

        return BooleanAlgebra.is_tautology(logic_min >> logic_tar)

    @staticmethod
    def in_downarrow(logic_max: Expr, logic_tar) -> bool:
        """
        Determine if downarrow({logic_max}) contains logic_tar

        Args:
            logic_max (Expr): the arg of downarrow
            logic_tar (Expr): the logical expression you want to know if contained or not.
        Returns:
            bool: True if contained.
        """

        assert (not isinstance(logic_max, list))
        assert (logic_max is not None)

        return BooleanAlgebra.is_tautology(logic_tar >> logic_max)

    @staticmethod
    def is_included(smaller: list[Expr], bigger: list[Expr]):
        """
        Determine if a smaller is included in the bigger. smaller ≦ bigger?

        Args:
            smaller (list[Expr]): left-hand side
            bigger (list[Expr]): right-hand side

        Returns:
            bool: True if included.
        """

        for logic_s in smaller:

            included = False

            for logic_b in bigger:

                if BooleanAlgebra.is_equivalent(logic_s, logic_b):
                    included = True
                    break

            if not included:
                return False

        return True

    @staticmethod
    def is_boolean_algebra(logics: list[Expr]):
        """determine if logics is a complete boolean algebra.

        Args:
            logics (list[Expr]): the set of logical expressions

        Returns:
            bool: True if logics is a complete boolean algebra.
        """

        if len(logics) == 0:
            raise NotImplementedError()

        # determine if the Expr list logics contains S.true and S.false
        top_exist = False
        bot_exist = False

        for logic in logics:

            if BooleanAlgebra.is_equivalent(logic, S.true):
                top_exist = True

            if BooleanAlgebra.is_equivalent(logic, S.false):
                bot_exist = True

            if top_exist and bot_exist:
                break

        if (not top_exist) or (not bot_exist):
            return False

        # determine if the Expr list logics is closed under the NOT operation.
        for logic in logics:

            not_logic = ~ logic

            is_included = BooleanAlgebra.is_included([not_logic], logics)

            if not is_included:
                return False

        # determine if the Expr list logics is closed under the AND operation.
        for logic1, logic2 in itertools.permutations(logics, 2):

            and_logic = logic1 & logic2

            is_included = BooleanAlgebra.is_included([and_logic], logics)

            if not is_included:
                return False

        # determine if the Expr list logics is closed under the OR operation.
        for logic1, logic2 in itertools.permutations(logics, 2):

            or_logic = logic1 | logic2
            is_included = BooleanAlgebra.is_included([or_logic], logics)

            if not is_included:
                return False

        return True

    def gen_random_expr(self) -> Expr:

        num_propvar = self.NUM_PROPVAR

        # the right side of truth table.
        valuations_expr = [random.randint(0, 1)
                           for i in range(2 ** num_propvar)]

        clauses: list[Expr] = [S.false]

        # the left side of truth table representing props.
        valuations_props = itertools.product(*([[0, 1]] * num_propvar))

        # for each rows of the truth table of the expr
        for rowi, valuation_props in enumerate(valuations_props):
            # print(rowi)
            valuation_expr = valuations_expr[rowi]

            if valuation_expr == 0:
                continue

            clause: Expr = sympy.S.true

            for propi in range(num_propvar):

                if valuation_props[propi] == 1:
                    clause = (clause) & (self.PROPVARS[propi])
                else:
                    clause = (clause) & (~self.PROPVARS[propi])

            clauses.append(clause)

        expr: Expr = S.false

        for clause in clauses:
            expr = (expr) | (clause)

        return expr

    def visualize(self, title="boolean-algebra", path_to_save_dir="./"):
        """
        visualize the boolean algebra.

        Args:
            title (str, optional):filename. Defaults to "boolean-algebra".
            path_to_save_dir (str, optional): path to directory you want to save pdf in. Defaults to "./".
        """

        UNIV_SET: list[Expr] = []

        num_propvar = self.NUM_PROPVAR

        def construct_exprs() -> Generator[Expr, None, None]:
            """
            generate the elems of the BooleanAlgebra

            Yields:
                Generator[pyprover.logic.Expr,None,None]: an elem in the BooleanAlgebra
            """

            # the all patterns of the right side of truth table.
            valuations_exprs = itertools.product(
                *([[0, 1]] * (2 ** num_propvar)))

            # for each expr
            for valuations_expr in valuations_exprs:
                clauses = [S.false]

                # the left side of truth table representing props.
                valuations_props = itertools.product(*([[0, 1]] * num_propvar))

                # for each rows of the truth table of the expr
                for rowi, valuation_props in enumerate(valuations_props):
                    valuation_expr = valuations_expr[rowi]

                    if valuation_expr == 0:
                        continue

                    clause = S.true

                    for propi in range(num_propvar):

                        if valuation_props[propi] == 1:
                            clause = (clause) & (self.PROPVARS[propi])
                        else:
                            clause = (clause) & (~self.PROPVARS[propi])

                    clauses.append(clause)

                expr = S.false

                for clause in clauses:
                    expr = (expr) | (clause)

                yield expr  # pyprover.simplest_form(pyprover.dnf(expr))

        for expr in construct_exprs():
            expr = simplify_logic(expr)
            UNIV_SET.append(expr)

        hasse = nx.DiGraph()
        hasse.add_nodes_from(UNIV_SET)

        for node1 in hasse.nodes:
            for node2 in hasse.nodes:
                if node1 is node2:
                    continue

                if BooleanAlgebra.is_tautology(node1 >> node2):
                    hasse.add_edge(node1, node2)

        ALL_EDGES = list(hasse.edges)

        for e in ALL_EDGES:
            u, v = e[0], e[1]
            assert (u is not v)
            hasse.remove_edge(u, v)
            reachable = (v in nx.descendants(hasse, u))
            if reachable:
                continue
            else:
                hasse.add_edge(u, v)

        # setting output size
        hasse.graph['size'] = "70.75,100.25"
        hasse.graph["rankdir"] = "BT"

        # Convert this to agraph class (PyGraphviz)
        G_pgv = nx.nx_agraph.to_agraph(hasse)
        # print(G_pgv.nodes()) # Vertex name was changed to str by conversion.

        # labelにvis_infoを入れてやることで情報を可視化できるようにする。
        for node in G_pgv.nodes():
            G_pgv.get_node((node)).attr["label"] = str(node)

        # Optional prog=[‘neato’|’dot’|’twopi’|’circo’|’fdp’|’nop’] will use specified graphviz layout method.
        # fdp is recommended.
        # ValueError: Program osage is not one of: neato, dot, twopi, circo, fdp, nop, gc, acyclic, gvpr, gvcolor, ccomps, sccmap, tred, sfdp, unflatten.
        G_pgv.draw(path_to_save_dir+f"{title}.pdf",
                   prog='dot', args='-Gnodesep=1')

        return

    def __repr__(self) -> str:

        return str(self.PROPVARS)

    @staticmethod
    def calc_meet(logics: list[Expr]) -> list[Expr]:
        """
        Calculate the meet(AND) of the given logical expressions.

        If no logical expressions are given, this will return an empty list. 

        This returns a list including only one logical expression otherwise.

        Args:
            logics (list[pyprover.Expr]): logical expressions. 

        Returns:
            list[pyprover.Expr]: the meet of the logical expressions.
        """

        if len(logics) == 0:
            return []

        meet = S.true

        for logic in logics:
            meet = meet & logic

        return [meet]

    @staticmethod
    def calc_join(logics: list[Expr]) -> list[Expr]:
        """
        Calculate the join(OR) of the given logical expressions.

        If no logical expressions are given, this will return an empty list. 

        This returns a list including only one logical expression otherwise.

        Args:
            logics (list[pyprover.Expr]): logical expressions. 

        Returns:
            list[pyprover.Expr]: the join of the logical expressions.
        """

        if len(logics) == 0:
            return []

        join = S.false

        for logic in logics:
            join = join | logic

        return [join]

    @staticmethod
    def _calc_minimal_representations(D: set[Expr]) -> Union[list[set[Expr]], set[list]]:
        """
        calculate the minimal representations of set D.


        Args:
            D (set[Expr]): set of logical expressions.

        Returns:
            list[set[Expr]]: minimal representations of D. If D is an empty set, this returns set([]).  
        """

        minimal_representations: list[set[Expr]] = []
        smaller_set_in_prec: list[set[Expr]] = []

        # If D is an empty set, no strict subsets of D exist.
        if D == set([]):
            return set([])

        meet_D = BooleanAlgebra.calc_meet(list(D))[0]

        for D2 in powerset(D):  # type: ignore
            D2 = set(D2)

            if D2 == set([]):
                continue

            meet_D2 = BooleanAlgebra.calc_meet(list(D2))[0]

            if not BooleanAlgebra.is_equivalent(meet_D, meet_D2):
                continue

            # here, found out D2 \prec

            smaller_set_in_prec.append(D2)

        for target in smaller_set_in_prec:

            is_smaller = True

            for rival in smaller_set_in_prec:

                if (not BooleanAlgebra.is_included(list(rival), list(target))) or BooleanAlgebra.is_included(list(target), list(rival)):

                    continue

                is_smaller = False
                break

            if is_smaller:
                minimal_representations.append(target)

        return minimal_representations

    @staticmethod
    def prec1(D1: set[Expr], D2: set[Expr]) -> bool:
        """
        Determine whether or not D1 \prec D2 or not. D2 is bigger than or equal to D1, this returns True.
        This \\prec is used in the definition of minimal representation.

        Args:
            D1 (set[Expr]): left-hand side
            D2 (set[Expr]): right-hand side

        Returns:
            bool: D2 is bigger than or equal to D1, this returns True.
        """

        is_included = BooleanAlgebra.is_included(list(D1), list(D2))
        if not is_included:
            return False

        meet_D1 = BooleanAlgebra.calc_meet(list(D1))
        meet_D2 = BooleanAlgebra.calc_meet(list(D2))

        if meet_D1 == [] and meet_D2 == []:
            return True
        elif meet_D1 == [] or meet_D2 == []:
            return False

        return BooleanAlgebra.is_equivalent(meet_D1[0], meet_D2[0])


class Interpretation():
    def __init__(self, typed_graph: TypedGraph, D: BooleanAlgebra, limit_image_size: int = 5):
        """
        Interpretation

        Args:
            typed_graph (TypedGraph): typed graph
            D (BooleanAlgebra): boolean algebra
            limit_image_size (int, optional): the maximum number of elements in the output of Interpretation. Defaults to 5.

        """

        count = 0

        self.OMEGA = "omega"
        self.Themes = typed_graph.Themes
        self.A = typed_graph.Aord + typed_graph.Apnt

        # print(self.A)
        # print(self.Themes)

        # It is assumed that tuple[str] in key of self.mapping is sorted.
        self.mapping: dict[tuple[tuple[str], str], list[Expr]] = dict()

        def power_set_themes() -> tuple[str]:

            s = list(self.Themes)
            ret = itertools.chain.from_iterable(
                itertools.combinations(s, r) for r in range(len(s) + 1))
            return cast(tuple[str], ret)

        # setting the mapping (subthemes,a) -> logic_exprs
        for subthemes in power_set_themes():
            assert (isinstance(subthemes, tuple))
            for a in self.A + [self.OMEGA]:
                # setting (subtheme,a) -> image

                image_size = random.randint(0, limit_image_size)
                image: list[Expr] = []

                for i in range(image_size):

                    count += 1
                    # print(count)

                    expr = D.gen_random_expr()
                    is_already_generated = False

                    for already_generated in image:

                        if already_generated == expr:
                            is_already_generated = True
                            break

                    if not is_already_generated:
                        image.append(expr)

                self.mapping[(subthemes, a)] = image

        # print(count)
        return

    def __repr__(self) -> str:
        return pprint.pformat(self.mapping)

    def __str__(self) -> str:
        return pprint.pformat(self.mapping)


class TAAMModel():
    def __init__(self, typed_graph: Optional[TypedGraph] = None, D: Optional[BooleanAlgebra] = None, I: Optional[Interpretation] = None):
        """Initialize the TAAMModel object.

        Args:
            typed_graph (Optional[TypedGraph], optional): a typed graph. Defaults to None.
            D (Optional[BooleanAlgebra], optional): a boolean algebra. Defaults to None.
            I (Optional[Interpretation], optional): an interpretation. Defaults to None.
        """

        if typed_graph is None:
            self.typed_graph = TypedGraph(3, 3, 1, 2, 2, 3)
        else:
            self.typed_graph = cast(TypedGraph, typed_graph)

        if D is None:
            self.D = BooleanAlgebra(2)
        else:
            self.D = cast(BooleanAlgebra, D)

        if I is None:
            self.I = Interpretation(self.typed_graph, self.D)
        else:
            self.I = cast(Interpretation, I)

        return

    def visualize(
            self,
            description="",
            title="result",
            path_to_save_dir="./",
            node_vis_features: list[str] = ['typed'],
            edge_vis_features: list[str] = ['themes'],
            add_description: bool = True):
        """
        Display networkx directed graphs (MultiDiGraph).
        It supports the display of attributes.
        Args:
            notes (str, optional): [description]. Defaults to "".
            title (str, optional): [title to save]. Defaults to "result".
            path_to_save_dir (str, optional): [destination path]. Defaults to "generated_pic/".
            vis_features(list[str],optional): Select attributes to display
            add_description(bool,optional):Whether to add a dummy vertex for description
        """
        graph: nx.DiGraph = self.typed_graph.graph

        num_vertex = len(list(graph.nodes))

        def attach_scc_id() -> tuple[nx.DiGraph, int]:
            """attach scc_id to nx.Digraph and nx.MultiDigraph

            Attach scc_id (same value for same scc) as an attribute to the graph.
            Returns:
                tuple[nx.DiGraph,int]: Graph after scc_id is assigned, Number of SCCs.
            """
            assert (isinstance(graph, nx.MultiDiGraph)
                    or isinstance(graph, nx.DiGraph))

            scc_id = 1

            for comp in sorted(nx.strongly_connected_components(graph), key=len, reverse=True):
                for node in comp:
                    graph.nodes[node]["scc_id"] = scc_id
                scc_id += 1

            return graph, scc_id-1

        # SCC
        graph, num_scc_group = attach_scc_id()

        def set_node_visinfo(node: int, attributes: list[str] = node_vis_features):
            """
            A function that attaches info to each vertex for use in visualization. 
            Args:
                node (int): a natural number representing a vertex.
                attributes (list[str], optional): key of information to visualize. Defaults to ['skew_type','label','conditions','predicted_labels','scc_id'].
            """

            info = ""

            info = f"node_index:{node}\n"

            for attr in attributes:

                try:

                    info += f"{attr}:{graph.nodes[node][attr]}\n"

                except:
                    pass

            graph.nodes[node]['vis_info'] = info
            graph.nodes[node]["fontsize"] = 15

            return

        def set_edge_visinfo(e: tuple[str, str], attributes: list[str] = edge_vis_features):
            """
            A function that attaches info to each vertex for use in visualization. 
            Args:
                node (int): a natural number representing a vertex.
                attributes (list[str], optional): key of information to visualize. Defaults to ['skew_type','label','conditions','predicted_labels','scc_id'].
            """

            info = ""

            for attr in attributes:

                try:

                    info += f"{attr}:{graph.edges[e][attr]}\n"

                except:
                    pass

            graph.edges[e]['vis_info'] = info
            graph.edges[e]["fontsize"] = 7

            return

        for node in list(graph.nodes):

            set_node_visinfo(node=node)

        for e in graph.edges():
            set_edge_visinfo(e)

        cm_name = 'jet'
        cm = plt.get_cmap(cm_name, num_scc_group+1)

        for node in list(graph.nodes):
            # 0-1 rgb
            color_rgb = cm(graph.nodes[node]['scc_id'])[:3]
            # hexadecimal rgb
            graph.nodes[node]['color'] = colors.to_hex(color_rgb)
            graph.nodes[node]['penwidth'] = 3

        graph.graph['overlap'] = "prism10000"

        # fdp,sfdp param
        graph.graph['K'] = 1.9
        # sfdp param
        graph.graph['repulsiveforce'] = 1.4

        # convert this into agraph（PyGraphviz）
        G_pgv = nx.nx_agraph.to_agraph(graph)

        for node in G_pgv.nodes():
            G_pgv.get_node(
                node).attr["label"] = graph.nodes[(node)]["vis_info"]

        for u, v in G_pgv.edges():
            G_pgv.get_edge(
                u, v).attr['label'] = f'<<table border="0" cellborder="0"><tr><td bgcolor="gray">{graph.edges[u,v]["vis_info"]}</td></tr></table>>'
            G_pgv.get_edge(
                u, v).attr['label'] = f'{graph.edges[u,v]["vis_info"]}'

        # Dummy vertices added to illustrate the generated image
        if add_description:
            G_pgv.graph_attr["label"] = description

        # either fdp or sfdp is recommended.
        G_pgv.draw(path_to_save_dir+f"{title}.pdf", prog='fdp',
                   args='-Gnodesep=1 -Gsize=100 -Gdpi=1000 -Gratio=0.6')

        return

    def __repr__(self) -> str:
        return str(self.typed_graph) + "\n" + str(self.D) + "\n" + str(self.I)

    def __str__(self) -> str:
        return str(self.typed_graph) + "\n" + str(self.D) + "\n" + str(self.I)

    def save_model(self, path="./model.dill"):
        """save TAAMModel into .dill file.

        Args:
            path (str, optional): a file path. Defaults to "./model.dill".
        """

        with open(path, "wb") as f:
            dill.dump(self, f)

        return

    @staticmethod
    def load_model(path="./model.dill") -> TAAMModel:
        """load TAAMModel from .dill file

        Args:
            path (str, optional): a file path. Defaults to "./model.dill".

        Raises:
            RuntimeError: Errors occurring when loading models

        Returns:
            TAAMModel: Loaded model
        """

        with open(path, "rb") as f:
            try:
                model: TAAMModel = dill.load(f)
            except Exception as e:
                traceback.print_exc()
                sys.exit()

        ret = TAAMModel()

        ret.typed_graph.Aord = model.typed_graph.Aord
        ret.typed_graph.Themes = model.typed_graph.Themes
        ret.typed_graph.rel = model.typed_graph.rel
        ret.typed_graph.Apnt = model.typed_graph.Apnt
        ret.typed_graph.graph = model.typed_graph.graph

        ret.D.NUM_PROPVAR = model.D.NUM_PROPVAR
        ret.D.PROPVARS = model.D.NUM_PROPVAR

        ret.I.OMEGA = model.I.OMEGA
        ret.I.Themes = model.I.Themes
        ret.I.A = model.I.A
        ret.I.mapping = model.I.mapping

        # determine if the model is proper

        for node in ret.typed_graph.enumerate_pnode("t.a"):
            a = node[node.find('.') + 1:]
            if a not in ret.typed_graph.graph.nodes:
                raise RuntimeError(
                    "contained a pnode that does not follow the format t.a where a in ONODE of the graph")

        return ret

    def calc_ws_set(self, t: str, s: str) -> list[set[str]]:
        """
        Given t in themes, s in nodes of the graph, enumerate the ws set ws(t,s).

        Args:
            t (str): t in self.typedgraph.Themes .
            s (str): s in self.typedgraph.graph.nodes .

        Returns:
            list[set[str]]: The list of ws set.
        """

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph

        if t not in graph.nodes[s]["themes"]:
            return []

        preds_attack: list[str] = []
        preds_support: list[str] = []

        for pred in graph.predecessors(s):

            if t not in graph.nodes[pred]["themes"]:
                continue

            if "attack" in graph.edges[pred, s]["rel"] and t in graph.edges[pred, s]["themes"]:
                preds_attack.append(pred)

            if "support" in graph.edges[pred, s]["rel"] and t in graph.edges[pred, s]["themes"]:
                preds_support.append(pred)

        maximal_attack = set(preds_attack + [s])
        maximal_support = set(preds_support + [s])

        if len(maximal_attack) > len(maximal_support):
            return [maximal_attack]
        elif len(maximal_support) > len(maximal_attack):
            return [maximal_support]
        else:
            return [maximal_attack, maximal_support] if maximal_attack != maximal_support else [maximal_attack]

    def calc_ds_set(self, t: str, s: str) -> list[set[str]]:
        """
        Given t in themes, s in nodes of the graph, enumerate the ws set ds(t,s).

        Args:
            t (str): t in self.typedgraph.Themes .
            s (str): s in self.typedgraph.graph.nodes .

        Returns:
            list[set[str]]: The list of ds set.
        """

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph

        if t not in graph.nodes[s]["themes"]:
            return []

        def calc_ds_for_rel(rel: str) -> list[set[str]]:

            assert (rel in ["attack", "support"])

            # remove redundant edges and nodes.
            simplified_graph = graph.copy()

            for e in graph.edges:
                if t in graph.edges[e]["themes"] and rel in graph.edges[e]["rel"]:
                    continue
                simplified_graph.remove_edge(*e)

            for node in graph.nodes:
                if t in graph.nodes[node]["themes"]:
                    continue
                simplified_graph.remove_node(node)

            # remove nodes unreachable to s

            reachable = nx.ancestors(simplified_graph, s)
            for node in list(simplified_graph.nodes):
                if node == s:
                    continue
                if node not in reachable:
                    simplified_graph.remove_node(node)

            for e in simplified_graph.edges:
                assert (rel in simplified_graph.edges[e]["rel"])
                assert (t in simplified_graph.edges[e]["themes"])

            condensation: nx.DiGraph = nx.condensation(simplified_graph)

            order: list[int] = list(nx.topological_sort(condensation))
            # The maximum number of vertices of the original graph that can be traced by the time it leaves the strongly connected component.
            num_vertices_walked = [-100] * len(order)
            # From which strongly connected component did the transition occur?
            prevs: list[list[int]] = [[]] * len(order)
            # i-th index contains the size of the scc of the node i in condensation graph.
            scc_size = [len(condensation.nodes[node]["members"])
                        for node in sorted(list(condensation.nodes))]

            assert (s in condensation.nodes[order[-1]]["members"])
            # print(s,t,rel)
            # print(condensation.nodes)
            # print(simplified_graph.nodes)
            # print(order)
            # print(condensation.nodes.data())

            # nx.draw(condensation)
            # plt.show()

            # dynamic programming on the DAG condensation
            for node in order:  # here, node denotes one of the strongly connected components.
                # print(node,num_vertices_walked)

                # initialization of the dp table.
                if num_vertices_walked[node] < 0:
                    num_vertices_walked[node] = scc_size[node]
                    assert (prevs[node] == [])
                    prevs[node] = [-1]

                for oute in condensation.out_edges(node):
                    assert (oute[0] == node)
                    # print(oute)
                    next = oute[1]
                    candidate = num_vertices_walked[node] + scc_size[next]

                    # Already found a better walk.
                    if candidate < num_vertices_walked[next]:
                        continue
                    # found one that walks the same number of vertices, so add it to the solution.
                    elif candidate == num_vertices_walked[next]:
                        prevs[next].append(node)
                    else:  # found a better walk.
                        num_vertices_walked[next] = candidate
                        prevs[next] = [node]

            # print(f"maximum :{num_vertices_walked[order[-1]]}")

            # restore the path way.

            # optimal walk on the condensation graph
            optimal_walks_on_condensation: list[list[int]] = []

            def dfs(nowpos: int, now_walk: list[int]):
                assert (isinstance(nowpos, int))

                if nowpos == -1:
                    optimal_walks_on_condensation.append(now_walk)
                    return

                for prev in prevs[nowpos]:
                    dfs(prev, now_walk + [nowpos])

                return

            dfs(order[-1], [])

            # print(f"optimal walk on scc: {optimal_walks_on_condensation}")
            assert (len(optimal_walks_on_condensation) > 0)

            # change the node index og the condensation graph to the nodes of the original graph
            ret_sets: list[set[str]] = []

            optimal_elems_num = num_vertices_walked[order[-1]]

            for walk_on_condensation in optimal_walks_on_condensation:
                restored_original_nodes: set[str] = set([])

                for sccid in walk_on_condensation:
                    nodes_in_the_scc: set[str] = set(
                        condensation.nodes[sccid]["members"])
                    restored_original_nodes = restored_original_nodes | nodes_in_the_scc

                ret_sets.append(restored_original_nodes)
                # print(f"calculating restored:{restored_original_nodes}")
                assert (len(restored_original_nodes) == optimal_elems_num)

            # print(f"ds_{rel}:{ret_sets}")
            return ret_sets

        ds_attack = calc_ds_for_rel(rel="attack")
        ds_support = calc_ds_for_rel(rel="support")

        def remove_duplicated(L: list[set]):
            seen = []
            for elem in L:
                if elem in seen:
                    continue
                seen.append(elem)
            return seen

        assert (ds_attack == remove_duplicated(ds_attack))
        assert (ds_support == remove_duplicated(ds_support))

        if ds_attack == [] and ds_support == []:
            return []
        elif ds_attack == []:
            return ds_support
        elif ds_support == []:
            return ds_attack
        else:
            size_attack = len(ds_attack[0])
            size_support = len(ds_support[0])

            if size_attack > size_support:
                return ds_attack
            elif size_support > size_attack:
                return ds_support
            else:

                res = ds_attack + ds_support
                res = remove_duplicated(res)

                return res

    def is_redundant(self, nodes: set[str], verbose: bool = False) -> bool:
        """
        For S (a subset of nodes of the graph), this judges whether or not it is redundant.

        Args:
            S (set[str]): a subset of nodes of the graph.
            verbose(bool,optional) : if True, output the overlapped minimal representation.

        Returns:
            bool: True if redundant, else False
        """

        # We can perform a Brute-force search for s1, s2, and T and see if there is a minimal representation that overlaps.
        assert (self.typed_graph is not None)
        Themes: list[str] = self.typed_graph.Themes

        for s1 in nodes:
            for s2 in nodes:

                if s1 == s2:
                    continue

                for T in powerset(Themes):

                    logics1 = set(self.I.mapping[(T, s1)])
                    logics2 = set(self.I.mapping[(T, s2)])

                    minimal_representations1 = BooleanAlgebra._calc_minimal_representations(
                        logics1)
                    if minimal_representations1 == set([]):
                        continue
                    minimal_representations2 = BooleanAlgebra._calc_minimal_representations(
                        logics2)
                    if minimal_representations2 == set([]):
                        continue

                    # judge if the two minimal representations intersect or not.

                    for representation1 in minimal_representations1:
                        for representation2 in minimal_representations2:

                            if BooleanAlgebra.is_included(list(representation1), list(representation2)) and  \
                                    BooleanAlgebra.is_included(list(representation2), list(representation1)):

                                if verbose:
                                    print(s1, s2, T)
                                    print(f"logics1:{logics1}")
                                    print(f"logics2:{logics2}")
                                    print(representation1, representation2)

                                # has intersection.
                                return True

        # for all combinations of s1,s2,T, there are not any overlapped minimal representations.
        return False

    def meet_tr(self) -> bool:

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph

        for e in graph.edges:
            u, v = e
            u = cast(str, u)
            v = cast(str, v)
            common = set(graph.edges[e]["typed"]) & set(
                graph.nodes[u]["typed"]) & set(graph.nodes[v]["typed"])

            if common == set():
                return False

        return True

    def meet_nnp(self) -> bool:

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph

        meet_constraint1 = True
        # for s = t.c
        for s in graph.nodes:
            s = cast(str, s)

            if s[-1] != "c":
                continue

            t = s[:s.find('.')]

            exists = False

            for s2 in graph.nodes:
                if t in graph.nodes[s2]["typed"]:
                    exists = True

            if not exists:
                meet_constraint1 = False
                break

            del exists

        if not meet_constraint1:
            return False

        meet_constraint2 = True
        # for s == t.a
        for s in graph.nodes:
            s = cast(str, s)
            if s[0] != 't' or s[-1] == 'c':
                continue

            t = s[:s.find('.')]
            a = s[s.find('.')+1:]

            if a not in list(graph.nodes):
                continue

            if t not in graph.nodes[a]['typed']:
                meet_constraint2 = False
                break

        return meet_constraint2

    def meet_nsa(self) -> bool:

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph

        for s in graph.nodes:

            if not graph.has_edge(s, s):
                continue

            if "attack" in graph.edges[s, s]['rel']:
                return False

        return True

    def meet_kos(self) -> bool:

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph

        # for s = t.a
        for s in graph.nodes:
            s = cast(str, s)

            if s[0] != 't' or s[-1] == 'c':
                continue

            t = s[:s.find('.')]
            a = s[s.find('.')+1:]

            if not graph.has_node(a):
                continue

            if not (set(graph.nodes[s]['typed']) <= set(graph.nodes[a]['typed'])):
                return False

        return True

    def meet_nss(self) -> bool:

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph

        for e in graph.edges:

            if "attack" not in graph.edges[e]["rel"]:
                continue

            if "support" in graph.edges[e]["rel"]:
                return False

            # (s,si),(s,sj)
            for s in graph.nodes:
                s = cast(str, s)

                for si, sj in [(e[0], e[1]), (e[1], e[0])]:
                    si = cast(str, si)
                    sj = cast(str, sj)

                    if not (graph.has_edge(s, si)) or ("support" not in graph.edges[s, si]["rel"]):
                        continue

                    if not graph.has_edge(s, sj):
                        continue

                    if "support" not in graph.edges[s, sj]["rel"]:
                        continue

                    if set(graph.edges[s, si]['themes']) & set(graph.edges[s, sj]['themes']) == set([]):
                        continue

                    return False

            # reversed version (si,s),(sj,s)
            for s in graph.nodes:
                s = cast(str, s)

                for si, sj in [(e[0], e[1]), (e[1], e[0])]:
                    si = cast(str, si)
                    sj = cast(str, sj)

                    if not (graph.has_edge(si, s)) or ("support" not in graph.edges[si, s]["rel"]):
                        continue

                    if not graph.has_edge(sj, s):
                        continue

                    if "support" not in graph.edges[sj, s]["rel"]:
                        continue

                    if set(graph.edges[si, s]['themes']) & set(graph.edges[sj, s]['themes']) == set([]):
                        continue

                    return False

        return True

    def meet_aass(self) -> bool:

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph

        for e in graph.edges:

            themes_e = graph.edges[e]["themes"]
            rel_e = graph.edges[e]["rel"]
            s1, s2 = e

            if "attack" in rel_e:

                for subthemes in powerset(themes_e):

                    if subthemes == ():
                        continue

                    logics_s1 = self.I.mapping[(subthemes, s1)]
                    logics_s2 = self.I.mapping[(subthemes, s2)]

                    inf1: list[Expr] = (BooleanAlgebra.calc_meet(logics_s1))
                    inf2: list[Expr] = (BooleanAlgebra.calc_meet(logics_s2))

                    if set(inf1) == set([]):
                        return False

                    if set(inf1) == set(inf2):
                        return False

                    if set(inf2) == set([]):
                        return False

            if "support" in rel_e:

                for subthemes in powerset(themes_e):

                    if subthemes == ():
                        continue

                    logics_s1 = self.I.mapping[(subthemes, s1)]
                    logics_s2 = self.I.mapping[(subthemes, s2)]

                    inf1 = BooleanAlgebra.calc_meet(logics_s1)
                    inf2 = BooleanAlgebra.calc_meet(logics_s2)

                    if set(inf1) == set([]):
                        return False
                    if set(inf2) == set([]):
                        return False

                    is_in_uparrow: bool = BooleanAlgebra.in_uparrow(
                        inf1[0], inf2[0])
                    is_in_downarrow: bool = BooleanAlgebra.in_downarrow(
                        inf1[0], inf2[0])

                    if (not is_in_uparrow) and (not is_in_downarrow):
                        return False

        return True

    def meet_i(self) -> bool:

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph
        Themes = self.typed_graph.Themes

        for s in graph.nodes:
            s = cast(str, s)

            for subthemes in powerset(Themes):

                subthemes = cast(tuple[str], subthemes)

                logics_left = self.I.mapping[(subthemes, s)]
                logics_right = self.I.mapping[(subthemes, self.I.OMEGA)]

                for logic_left in logics_left:

                    is_contained = False

                    for logic_right in logics_right:

                        is_contained |= BooleanAlgebra.is_equivalent(
                            logic_left, logic_right)

                        if is_contained:
                            break

                    if not is_contained:
                        return False

        return True

    def meet_vi(self) -> bool:

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph

        for s in graph.nodes:

            empty: Any = ()

            if self.I.mapping[(empty, s)] != []:
                return False

        return True

    def meet_bat(self) -> bool:

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph

        for subthemes in powerset(self.typed_graph.Themes):

            logics = self.I.mapping[(subthemes, self.I.OMEGA)]

            if logics == []:
                continue

            is_complete_boolean_algebra = BooleanAlgebra.is_boolean_algebra(
                logics)

            if not is_complete_boolean_algebra:
                return False

        return True

    def meet_pr(self) -> bool:

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph

        for subthemes in powerset(self.typed_graph.Themes):
            subthemes = cast(tuple[str], subthemes)

            # s == a
            for onode in self.typed_graph.enumerate_onode():
                left = self.I.mapping[(subthemes, onode)]

                image_onode: set[str] = set(
                    graph.nodes[onode]["themes"])  # Π(onode)
                # uniquify themes and calc the intersection
                integrated: Any = set(subthemes) & image_onode
                integrated = list(integrated)
                integrated.sort()  # I.mapping assumes themes are sorted
                # the key of I.mapping is tuple[tuple[str],str]
                integrated = tuple(integrated)

                right = self.I.mapping[(integrated, onode)]

                if not BooleanAlgebra.is_included(left, right):
                    return False

            # s == t.a
            for pnode_ta in self.typed_graph.enumerate_pnode(form="t.a"):
                left = self.I.mapping[(subthemes, pnode_ta)]

                # almost same operation as the case s == a
                image_pnode_ta: set[str] = set(graph.nodes[pnode_ta]["themes"])
                theme = pnode_ta[0:pnode_ta.find(".")]
                integrated = set(subthemes) & (image_pnode_ta | set([theme]))
                integrated = list(integrated)
                integrated.sort()
                integrated = tuple(integrated)

                a = pnode_ta[pnode_ta.find(".") + 1:]
                assert (a.find("c") == -1)
                assert (pnode_ta == theme + "." + a)

                right = self.I.mapping[(integrated, a)]

                if not BooleanAlgebra.is_included(left, right):
                    return False

            # s == t.c
            for pnode_tc in self.typed_graph.enumerate_pnode(form="t.c"):
                left = self.I.mapping[(subthemes, pnode_tc)]
                assert (pnode_tc[-1] == "c")

                # almost same operation as the case s == a
                image_pnode_tc: set[str] = set(graph.nodes[pnode_tc]["themes"])
                theme = pnode_tc[0:pnode_tc.find(".")]
                integrated = set(subthemes) & (image_pnode_tc | set([theme]))
                integrated = list(integrated)
                integrated.sort()
                integrated = tuple(integrated)

                assert (pnode_tc == theme + "." + "c")

                right = self.I.mapping[(integrated, pnode_tc)]

                if not BooleanAlgebra.is_included(left, right):
                    return False

        return True

    def meet_mat(self) -> bool:

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph

        Themes = self.typed_graph.Themes

        for T2 in powerset(Themes):
            for T1 in powerset(T2):

                image_T1 = self.I.mapping[(T1, self.I.OMEGA)]
                image_T2 = self.I.mapping[(T2, self.I.OMEGA)]

                if not BooleanAlgebra.is_included(image_T1, image_T2):
                    return False

        return True

    def meet_manss(self) -> bool:

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph

        Themes = self.typed_graph.Themes

        special_nodes = list(self.typed_graph.enumerate_pnode("t.c"))

        for T2 in powerset(Themes):
            for T1 in powerset(T2):

                # for nodes that follow the form of "a" or "t.a".
                for node in graph.nodes:
                    if node in special_nodes:
                        continue

                    image_T1 = self.I.mapping[(T1, node)]
                    image_T2 = self.I.mapping[(T2, node)]

                    if not BooleanAlgebra.is_included(image_T1, image_T2):
                        return False

        return True

    def meet_ss(self) -> bool:

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph

        for node in graph.nodes:

            node_themes: list[str] = graph.nodes[node]["themes"]

            for T in powerset(node_themes):

                image = self.I.mapping[(T, node)]

                if image == []:
                    continue

                meet: Expr = BooleanAlgebra.calc_meet(image)[0]

                if BooleanAlgebra.is_equivalent(meet, S.true) or BooleanAlgebra.is_equivalent(meet, S.false):
                    return False

        return True

    def meet_esr(self) -> bool:

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph

        Themes = self.typed_graph.Themes

        for T1 in powerset(Themes):
            for T2 in powerset(Themes):

                common: set[Expr] = set(self.I.mapping[(T1, self.I.OMEGA)]) & set(
                    self.I.mapping[(T2, self.I.OMEGA)])

                # print(T1,T2,common)

                for node in self.typed_graph.enumerate_pnode("t.c"):

                    image_T1: set[Expr] = set(self.I.mapping[(T1, node)])
                    image_T2: set[Expr] = set(self.I.mapping[(T2, node)])

                    if not ((image_T1 & common) == (image_T2 & common)):
                        return False

        return True

    def meet_ensr(self) -> bool:

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph

        Themes = self.typed_graph.Themes

        for T1 in powerset(Themes):
            for T2 in powerset(Themes):

                common: set[Expr] = set(self.I.mapping[(T1, self.I.OMEGA)]) & set(
                    self.I.mapping[(T2, self.I.OMEGA)])

                # print(T1,T2,common)

                # for nodes that follows the form of "t.a"
                for node in self.typed_graph.enumerate_pnode("t.a"):

                    image_T1: set[Expr] = set(self.I.mapping[(T1, node)])
                    image_T2: set[Expr] = set(self.I.mapping[(T2, node)])

                    if not ((image_T1 & common) == (image_T2 & common)):
                        return False

        return True

    def meet_eos(self) -> bool:

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph

        Themes = self.typed_graph.Themes

        for T1 in powerset(Themes):
            for T2 in powerset(Themes):

                common: set[Expr] = set(self.I.mapping[(T1, self.I.OMEGA)]) & set(
                    self.I.mapping[(T2, self.I.OMEGA)])

                # print(T1,T2,common)

                # for nodes that follows the form of "a"
                for node in self.typed_graph.enumerate_onode():

                    image_T1: set[Expr] = set(self.I.mapping[(T1, node)])
                    image_T2: set[Expr] = set(self.I.mapping[(T2, node)])

                    if not ((image_T1 & common) == (image_T2 & common)):
                        return False

        return True

    def meet_das(self) -> bool:

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph

        Themes = self.typed_graph.Themes

        for u, v in graph.edges:
            # print(u,v)
            for T in powerset(graph.edges[u, v]["themes"]):
                # print(T)
                if T == ():
                    continue

                if "attack" in graph.edges[u, v]["rel"] and set(T) <= set(graph.edges[u, v]["themes"]):

                    meet_u: list[Expr] = BooleanAlgebra.calc_meet(
                        self.I.mapping[(T, u)])

                    if meet_u == []:
                        continue

                    meet_v = BooleanAlgebra.calc_meet(self.I.mapping[(T, v)])

                    if meet_v == []:
                        continue

                    if BooleanAlgebra.in_downarrow(meet_v[0], meet_u[0]) or BooleanAlgebra.in_uparrow(meet_v[0], meet_u[0]):
                        return False

                if "support" in graph.edges[u, v]["rel"] and set(T) <= set(graph.edges[u, v]["themes"]):

                    meet_u = BooleanAlgebra.calc_meet(self.I.mapping[(T, u)])
                    meet_v = BooleanAlgebra.calc_meet(self.I.mapping[(T, v)])

                    # print("judge support")

                    # print(meet_u,meet_v)

                    if meet_u == [] and meet_v == []:
                        continue
                    elif meet_u == [] or meet_v == []:
                        return False

                    if not BooleanAlgebra.is_equivalent(meet_u[0], meet_v[0]):
                        return False

        return True

    def meet_nwci(self):

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph

        for u, v in graph.edges:

            # print(u,v)

            if not "attack" in graph.edges[u, v]["rel"]:
                continue

            for T in powerset(graph.edges[u, v]["themes"]):
                # print(T)

                if T == ():
                    continue

                meet_u_list = BooleanAlgebra.calc_meet(self.I.mapping[(T, u)])
                meet_v_list = BooleanAlgebra.calc_meet(self.I.mapping[(T, v)])

                # print(meet_u_list,meet_v_list)

                # If meet_u_list is an empty set, the second condition is violated. The sets uparrow,downarrow does not include {empty set} in its value range.
                if meet_u_list == []:
                    return False

                # If meet_v is an empty set, the second condition is violated. uparrow and downarrow are empty sets, so meet_u is never included in the right-hand side.
                if meet_v_list == []:
                    return False

                meet_u = meet_u_list[0]
                meet_v = meet_v_list[0]

                # print(simplify_logic(meet_u),simplify_logic(meet_v))

                # check first condition
                # print("condition1")

                if (not BooleanAlgebra.in_uparrow(~ meet_v, meet_u)) or (BooleanAlgebra.is_equivalent(meet_u, ~ meet_v)):
                    pass
                else:
                    return False

                # check second condition
                # print("condition2")

                isin1 = BooleanAlgebra.in_downarrow(meet_v, meet_u)
                if isin1:
                    # print(1)
                    continue
                isin2 = BooleanAlgebra.in_uparrow(meet_v, meet_u)
                if isin2:
                    # print(2)
                    continue

                isin3 = BooleanAlgebra.in_downarrow(~ meet_v, meet_u)
                if isin3:
                    # print(3)
                    continue
                isin4 = BooleanAlgebra.in_uparrow(~ meet_v, meet_u)
                if isin4:
                    # print(4)
                    continue

                # did not satisfy the second condition.
                # print("not in for all")
                return False

        return True

    def meet_faD(self):

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph

        for s in graph.nodes:
            for t in graph.nodes[s]["themes"]:

                ds_sets = self.calc_ds_set(t, s)

                for ds_set in ds_sets:

                    ds_is_redundant = self.is_redundant(ds_set)

                    if ds_is_redundant:
                        return False

        return True

    def meet_faW(self):

        assert (self.typed_graph is not None)
        graph = self.typed_graph.graph

        for s in graph.nodes:
            for t in graph.nodes[s]["themes"]:

                ws_sets = self.calc_ws_set(t, s)

                for ws_set in ws_sets:

                    ws_is_redundant = self.is_redundant(ws_set)

                    if ws_is_redundant:
                        return False

        return True


def _main() -> int:  # pragma: no cover
    print("SEED:", SEED)

    random.seed(SEED)
    g = TypedGraph(Aord_size=4, Themes_size=1, num_pnode=2,
                   num_onode=2, num_edge=4, limit_num_given_themes=4)
    print(g)

    d = BooleanAlgebra(1)
    print(d)

    I = Interpretation(g, d, limit_image_size=4)

    # print(I)

    model = TAAMModel(g, d, I)

    print("====")

    print("G,PI")

    meet_constraint = dict()
    meet_constraint["tr"] = model.meet_tr()
    meet_constraint["nnp"] = model.meet_nnp()
    meet_constraint["nsa"] = model.meet_nsa()
    meet_constraint['kos'] = model.meet_kos()
    meet_constraint["nss"] = model.meet_nss()

    print("I")

    meet_constraint["aass"] = model.meet_aass()
    meet_constraint["i"] = model.meet_i()
    meet_constraint["vi"] = model.meet_vi()
    meet_constraint["bat"] = model.meet_bat()
    meet_constraint["pr"] = model.meet_pr()
    meet_constraint["mat"] = model.meet_mat()
    meet_constraint["manss"] = model.meet_manss()
    meet_constraint["ss"] = model.meet_ss()
    meet_constraint["esr"] = model.meet_esr()
    meet_constraint["ensr"] = model.meet_ensr()
    meet_constraint["eos"] = model.meet_eos()
    meet_constraint["das"] = model.meet_das()
    meet_constraint["nwci"] = model.meet_nwci()
    meet_constraint["faD"] = model.meet_faD()
    meet_constraint["faW"] = model.meet_faW()

    print("====")

    print(meet_constraint)

    ws_ds_result = dict()

    print("enum ws")

    for s in model.typed_graph.graph.nodes:
        for t in model.typed_graph.Themes:
            print(f"(s,t):{(s,t)}")
            r = model.calc_ws_set(t, s)
            ws_ds_result[(s, t, "ws")] = r

    print("enum ds")

    for s in model.typed_graph.graph.nodes:
        for t in model.typed_graph.Themes:
            print(f"(s,t):{(s,t)}")
            r = model.calc_ds_set(t, s)
            ws_ds_result[(s, t, "ds")] = r

    model.visualize(
        description=f"{pprint.pformat(ws_ds_result)}", title="ws_ds_set_not_tested")
    model.save_model(f"./ws_ds_set_not_tested.dill")

    print("redundant")

    redundant_S = []

    for nodes in powerset(model.typed_graph.graph.nodes):
        print(f" judging if {nodes} is redundant...")
        if model.is_redundant(set(nodes)):
            redundant_S.append(nodes)
            continue
        print(nodes, "not redundant", "!!!!!!!!!!")

    filename = "nottested"
    model.visualize(
        description=f"{model.typed_graph}\nconstraints :{meet_constraint}\nredundant S:{redundant_S}\nws and ds sets:{pprint.pformat(ws_ds_result)}\nI:{model.I}", title=filename)
    model.save_model(f"./{filename}.dill")

    print("faD:", model.meet_faD())
    print("faW:", model.meet_faW())

    return 0


if __name__ == "__main__":  # pragma: no cover

    _main()
