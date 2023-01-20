import sys
import time
import random

sys.path.append("./model_src")

from TAAMmodel import TAAMModel,BooleanAlgebra,Interpretation,TypedGraph


SEED = time.time()
random.seed(SEED)


class ConstraintChecker():

    constraints_names = [
        "tr",
        "nnp",
        "nsa",
        "kos",
        "nss",

        "aass",
        "i",
        "vi",
        "bat",
        "pr",
        "mat",
        "manss",
        "ss",
        "esr",
        "ensr",
        "eos",
        "das",
        "nwci",
        "faD",
        "faW"
    ]

    def __init__(self,model : TAAMModel,antecedent  ,consequent : list[str]) -> None:

        self.model = model
        self.antecedent = antecedent
        self.consequent = consequent

        return

    def check_constraint(self,constraint_name : str) -> bool:

        assert(constraint_name in ConstraintChecker.constraints_names)

        if constraint_name == "tr":
            return self.model.meet_tr()
        elif constraint_name == "nnp":
            return self.model.meet_nnp()
        elif constraint_name == "nsa":
            return self.model.meet_nsa()
        elif constraint_name == "kos":
            return self.model.meet_kos()
        elif constraint_name == "nss":
            return self.model.meet_nss()
        
        elif constraint_name == "aass":
            return self.model.meet_aass()
        elif constraint_name == "i":
            return self.model.meet_i()
        elif constraint_name == "vi":
            return self.model.meet_vi()
        elif constraint_name == "bat":
            return self.model.meet_bat()
        elif constraint_name == "pr":
            return self.model.meet_pr()
        elif constraint_name == "mat":
            return self.model.meet_mat()
        elif constraint_name == "manss":
            return self.model.meet_manss()
        elif constraint_name == "ss":
            return self.model.meet_ss()
        elif constraint_name == "esr":
            return self.model.meet_esr()
        elif constraint_name == "ensr":
            return self.model.meet_ensr()
        elif constraint_name == "eos":
            return self.model.meet_eos()
        elif constraint_name == "das":
            return self.model.meet_das()
        elif constraint_name == "nwci":
            return self.model.meet_nwci()
        elif constraint_name == "faD":
            return self.model.meet_faD()
        elif constraint_name == "faW":
            return self.model.meet_faW()
        
        else:
            print(constraint_name)
            assert(False)

    
    def run_checker(self):

        counter = 0

        # check antecedent
        for constraint_name in self.antecedent:
            result = self.check_constraint(constraint_name)
            counter += 1
            if not result:
                return True,counter
        
        for constraint_name in self.consequent:
            result = self.check_constraint(constraint_name)
            counter += 1
            if not result:
                return False,counter

        assert(counter == len(self.antecedent) + len(self.consequent))
        return True,counter
        
        

def run_a_single_test():
    model = TAAMModel()

    antecedent = ["aass","i","vi","bat","pr","mat","manss","ss",]
    # antecedent = []
    consequent = ["tr","nnp","nsa","kos","nss"]

    random.shuffle(antecedent)



    g = TypedGraph(Aord_size=6,Themes_size=3,num_pnode=1,num_onode=3,num_edge=8,limit_num_given_themes=4)

    d = BooleanAlgebra(1)

    I = Interpretation(g,d,limit_image_size=4)

    model = TAAMModel(g,d,I)


    
    print(model)
    checker = ConstraintChecker(model,antecedent,consequent)

    result,counter = checker.run_checker()


    if result:
        print("OK",counter)
        return True,model

    else:
        print("NG",counter)
        return False,model


if __name__ == "__main__":

    print("random test1 starts")

    for i in range(1,1000000):
        print(f"test{i}:")
        result,model = run_a_single_test()

        if not result:
            model.visualize(
                description=f"{model.typed_graph}\nI:{model.I}",
                title = f"counterexample{i}"
            )
            model.save_model(f"./counterexample{i}.dill")
            break


    