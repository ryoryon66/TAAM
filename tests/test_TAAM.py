#pytest
import pytest
from sympy import S
import sys
sys.path.append("./model_src")
sys.path.append("../model_src/")
sys.path.append("..")



from TAAMmodel import TAAMModel,BooleanAlgebra,TypedGraph,Interpretation, powerset



class TestTypedGraph(object):

    def test_enumerate_onode_and_pnode(self):

        typed_graph = TypedGraph(num_pnode=5,num_onode=3,num_edge = 4,limit_num_given_themes=1)

        onodes = []
        pnodes = []

        typed_graph.is_well_formed()

        for onode in typed_graph.enumerate_onode():
            onodes.append(onode)
            assert(onode.find("t") == -1)
        
        for pnode in typed_graph.enumerate_pnode():
            pnodes.append(pnode)
        
        for pnode in typed_graph.enumerate_pnode("t.a"):
            assert(pnode in pnodes)
            assert(pnode.find("c") == -1 )

        for pnode in typed_graph.enumerate_pnode("t.c"):
            assert(pnode in pnodes)
            assert(pnode.find("c") > -1 )
        
        with pytest.raises(Exception) as e:
            list(typed_graph.enumerate_pnode("c.t"))
        
        assert(str(e.value) == 'form should be "t.a", "t.c" or None')

        typed_graph.visualize(title="pytest")
        s = repr(typed_graph)

        assert(len(onodes) == 3)
        assert(len(pnodes) == 5)

        print("_" * 50)
        print(f"all nodes:{set(typed_graph.graph.nodes)}")
        print(f"onodes:{set(onodes)},pnodes:{set(pnodes)}")
        print("_" * 50)





class TestTAAMModel(object):

    def test_load_and_save(self):
        typed_graph = TypedGraph(Themes_size=4)
        D = BooleanAlgebra(2)
        I = Interpretation(typed_graph,D,2)

        path = "./tests/model_misc/model_save_load.dill"

        model = TAAMModel(typed_graph,D,I)
        model.typed_graph.is_well_formed()
        model.save_model(path)
        model = TAAMModel.load_model(path)
        model.save_model(path)
        model2 =TAAMModel.load_model(path)
        model2.save_model(path)
        model2.meet_nsa()
        model2.typed_graph.is_well_formed()
        model2.visualize(title="pytest")
        s = repr(model2)
        s = repr(model2.I)


        return
    
    def test_detect_improper_pnodes(self):

        base = "./tests/model_misc/"

        path1 = base + "model_improper_pnode1.dill"

        with pytest.raises(RuntimeError) as e:
            TAAMModel.load_model(path1)

        assert(str(e.value) == "contained a pnode that does not follow the format t.a where a in ONODE of the graph")

        return

    
    def test_calc_ws_set(self):

        print("_" * 50)
        print("test_calc_ws_set")

        base = "./constraints/ws_set/"

        path1 = base + "ws_ds_set1.dill"
        path2 = base + "ws_ds_set2.dill"

        model1 = TAAMModel.load_model(path1)
        model2 = TAAMModel.load_model(path2)

        def is_equal(L1 :list[set], L2: list[set]):

            print(L1,L2)

            for elem1 in L1:
                if elem1 not in L2:
                    return False
            
            for elem2 in L2:
                if elem2 not in L1:
                    return False
            
            return True
        
        assert(is_equal(model1.calc_ws_set("t1","0"),[{"0"}]) )
        assert(is_equal(model1.calc_ws_set("t3","0"),[]) )
        assert(is_equal(model1.calc_ws_set("t0","6"),[{'t1.1', '2', '6'}]) )


        assert(is_equal(model2.calc_ws_set("t2","3"),[{'t1.c', '3', '5', '2'}, {'t1.2', '3', '5', '2'}]) )    




        return

    def test_calc_ds_set(self):


        print("_" * 50)
        print("test_calc_ds_set")

        base = "./constraints/ds_set/"

        path1 = base + "ws_ds_set1.dill"
        path2 = base + "ws_ds_set2.dill"

        model1 = TAAMModel.load_model(path1)
        model2 = TAAMModel.load_model(path2)

        def is_equal(L1 :list[set], L2: list[set]):

            print(L1,L2)

            for elem1 in L1:
                if elem1 not in L2:
                    return False
            
            for elem2 in L2:
                if elem2 not in L1:
                    return False
            
            return True
        
        assert(is_equal(model1.calc_ds_set("t1","0"),[{"0"}]) )
        assert(is_equal(model1.calc_ds_set("t3","0"),[]) )
        assert(is_equal(model1.calc_ds_set("t0","6"),[{'1', '6', '4', '3', '2'}]) )

        assert(is_equal(model2.calc_ds_set("t2","t1.2"),[{'t1.2', '2', '5', '3'}, {'t1.2', 't1.c', '5', '3'}]))

        return 



    def test_is_redundant(self):

        base = "./constraints/is_redundant/"

        path1 = base + "is_redundant_model1.dill"
        path2 = base + "is_redundant_model2.dill"
        path3 = base + "is_redundant_model3.dill"
        path4 = base + "is_redundant_model4.dill"

        model1 = TAAMModel.load_model(path1)
        model2 = TAAMModel.load_model(path2)
        model3 = TAAMModel.load_model(path3)
        model4 = TAAMModel.load_model(path4)

        
        def is_equal(ans : list[set[str]] ,res : list[set[str]]):

            for s1 in ans:
                exists = False
                for s2 in res:
                    if s1 == s2:
                        exists = True
                        break
                
                if not exists:
                    return False

            for s1 in res:
                exists = False
                for s2 in ans:
                    if s1 == s2:
                        exists = True
                        break
                
                if not exists:
                    return False
        
            return True

        def test_model(model : TAAMModel,ans : list[set[str]]):
            
            res = []

            for nodes in powerset(model.typed_graph.graph.nodes):
                if model.is_redundant(set(nodes)):
                    res.append(set(nodes))
            
            return is_equal(res,ans)
        
        ans = [
            
        ]
        assert(test_model(model1,ans))


        ans = [
            
        ]
        assert(test_model(model2,ans))

        ans = [
            set(["1","2","3"])
        ]
        assert(not test_model(model2,ans))
        
        ans = [
            set(["2","t0.2"]),
        ]
        assert(test_model(model3,ans))


        ans = [
            set(["1","t0.1"]),
            set(["1","t0.1","t0.c"])
        ]
        assert(test_model(model4,ans))


        ans = [
            set(["1","t0.1","t0.c"]),
            set(["t0.1","1"]),
        ]
        assert(test_model(model4,ans))


        return 



    def test_meet_nnp(self):

        print("_" * 50)
        print("test_meet_nnp")

        base = "./constraints/nnp/"

        path1 = base + "model_nnp_False1.dill"
        path2 = base + "model_nnp_True1.dill"
        path3 = base + "model_nnp_False2.dill"
        path4 = base + "model_nnp_True2.dill"



        model1 = TAAMModel.load_model(path1)
        model2 = TAAMModel.load_model(path2)
        model3 = TAAMModel.load_model(path3)
        model4 = TAAMModel.load_model(path4)
        
        
        assert(not model1.meet_nnp())
        assert(model2.meet_nnp())
        assert(not model3.meet_nnp())
        assert(model4.meet_nnp())

        print("_" * 50)
        print("test_meet_nsa")

        base = "./constraints/nsa/"

        path1 = base + "model_nsa_False1.dill"
        path2 = base + "model_nsa_True1.dill"
        path3 = base + "model_nsa_False2.dill"
        path4 = base + "model_nsa_True2.dill"



        model1 = TAAMModel.load_model(path1)
        model2 = TAAMModel.load_model(path2)
        model3 = TAAMModel.load_model(path3)
        model4 = TAAMModel.load_model(path4)
        
        
        assert(not model1.meet_nsa())
        assert(model2.meet_nsa())
        assert(not model3.meet_nsa())
        assert(model4.meet_nsa())

        print("_" * 50)


    def test_meet_nsa(self):

        print("_" * 50)
        print("test_meet_nsa")

        base = "./constraints/nsa/"

        path1 = base + "model_nsa_False1.dill"
        path2 = base + "model_nsa_True1.dill"
        path3 = base + "model_nsa_False2.dill"
        path4 = base + "model_nsa_True2.dill"



        model1 = TAAMModel.load_model(path1)
        model2 = TAAMModel.load_model(path2)
        model3 = TAAMModel.load_model(path3)
        model4 = TAAMModel.load_model(path4)
        
        
        assert(not model1.meet_nsa())
        assert(model2.meet_nsa())
        assert(not model3.meet_nsa())
        assert(model4.meet_nsa())

        print("_" * 50)


        
    
    def test_meet_aass(self):

        base = "./constraints/aass/"

        path1 = base + "model_aass_False1.dill"
        path2 = base + "model_aass_True1.dill"



        model1 = TAAMModel.load_model(path1)
        model2 = TAAMModel.load_model(path2)
        
        
        assert(not model1.meet_aass())
        assert(model2.meet_aass())

    
    def test_meet_i(self):

        base = "./constraints/i/"

        path1 = base + "model_i_False1.dill"
        path2 = base + "model_i_True1.dill"



        model1 = TAAMModel.load_model(path1)
        model2 = TAAMModel.load_model(path2)
        
        
        assert(not model1.meet_i())
        assert(model2.meet_i())

        # Not visually checked.
        assert(model2.meet_tr())
        assert(model2.meet_nnp())
        assert(not model2.meet_nsa())
        assert(model2.meet_kos())
        assert(not model2.meet_nss())
        assert(not model2.meet_aass())
    

    def test_meet_vi(self):

        base = "./constraints/vi/"

        path1 = base + "model_vi_False1.dill"
        path2 = base + "model_vi_True1.dill"



        model1 = TAAMModel.load_model(path1)
        model2 = TAAMModel.load_model(path2)
        
        
        assert(not model1.meet_vi())
        assert(model2.meet_vi())



    def test_meet_bat(self):

        base = "./constraints/bat/"

        path1 = base + "model_bat_False1.dill"
        path2 = base + "model_bat_True1.dill"
        path3 = base + "model_bat_False2.dill"
        path4 = base + "model_bat_True2.dill"



        model1 = TAAMModel.load_model(path1)
        model2 = TAAMModel.load_model(path2)
        model3 = TAAMModel.load_model(path3)
        model4 = TAAMModel.load_model(path4)
        
        
        assert(not model1.meet_bat())
        assert(model2.meet_bat())
        assert(not model3.meet_bat())
        assert(model4.meet_bat())

        # Not visually checked.
        assert(model1.meet_tr())
        assert(model1.meet_nnp())
        assert(model1.meet_nsa())
        assert(model1.meet_kos())
        assert(not model1.meet_nss())

        assert(not model1.meet_aass())
        assert(not model1.meet_i())
        assert(not model1.meet_vi())


        assert(model2.meet_tr())
        assert(model2.meet_nnp())
        assert(not model2.meet_nsa())
        assert(model2.meet_kos())
        assert(not model2.meet_nss())

        assert(not model2.meet_aass())
        assert(not model2.meet_i())
        assert(not model2.meet_vi())



    def test_meet_pr(self):

        print("_" * 50)
        print("test_meet_pr")

        base = "./constraints/pr/"

        path1 = base + "model_pr_False1.dill"
        path2 = base + "model_pr_True1.dill"
        path3 = base + "model_pr_False2.dill"
        path4 = base + "model_pr_True2.dill"



        model1 = TAAMModel.load_model(path1)
        model2 = TAAMModel.load_model(path2)
        model3 = TAAMModel.load_model(path3)
        model4 = TAAMModel.load_model(path4)
        
        
        assert(not model1.meet_pr())
        assert(model2.meet_pr())
        assert(not model3.meet_pr())
        assert(model4.meet_pr())


        # Not visually checked.
        assert(model1.meet_tr())
        assert(model1.meet_nnp())
        assert(not model1.meet_nsa())
        assert(model1.meet_kos())
        assert(not model1.meet_nss())

        assert(not model1.meet_aass())
        assert(not model1.meet_i())
        assert(not model1.meet_vi())
        assert(not model1.meet_bat())


        assert(model2.meet_tr())
        assert(model2.meet_nnp())
        assert(not model2.meet_nsa())
        assert(model2.meet_kos())
        assert(not model2.meet_nss())

        assert(not model2.meet_aass())
        assert(not model2.meet_i())
        assert(not model2.meet_vi())
        assert(not model2.meet_bat())

        print("_" * 50)


    def test_meet_mat(self):

        print("_" * 50)
        print("test_meet_mat")

        base = "./constraints/mat/"

        path1 = base + "model_mat_False1.dill"
        path2 = base + "model_mat_True1.dill"
        path3 = base + "model_mat_False2.dill"
        path4 = base + "model_mat_True2.dill"



        model1 = TAAMModel.load_model(path1)
        model2 = TAAMModel.load_model(path2)
        model3 = TAAMModel.load_model(path3)
        model4 = TAAMModel.load_model(path4)
        
        
        assert(not model1.meet_mat())
        assert(model2.meet_mat())
        assert(not model3.meet_mat())
        assert(model4.meet_mat())

        print("_" * 50)




    def test_meet_manss(self):

        print("_" * 50)
        print("test_meet_manss")

        base = "./constraints/manss/"

        path1 = base + "model_manss_False1.dill"
        path2 = base + "model_manss_True1.dill"
        path3 = base + "model_manss_False2.dill"
        path4 = base + "model_manss_True2.dill"



        model1 = TAAMModel.load_model(path1)
        model2 = TAAMModel.load_model(path2)
        model3 = TAAMModel.load_model(path3)
        model4 = TAAMModel.load_model(path4)
        
        
        assert(not model1.meet_manss())
        assert(model2.meet_manss())
        assert(not model3.meet_manss())
        assert(model4.meet_manss())

        print("_" * 50)



    def test_meet_ss(self):

        print("_" * 50)
        print("test_meet_ss")

        base = "./constraints/ss/"

        path1 = base + "model_ss_False1.dill"
        path2 = base + "model_ss_True1.dill"
        path3 = base + "model_ss_False2.dill"
        path4 = base + "model_ss_True2.dill"



        model1 = TAAMModel.load_model(path1)
        model2 = TAAMModel.load_model(path2)
        model3 = TAAMModel.load_model(path3)
        model4 = TAAMModel.load_model(path4)
        
        
        assert(not model1.meet_ss())
        assert(model2.meet_ss())
        assert(not model3.meet_ss())
        assert(model4.meet_ss())

        print("_" * 50)





    def test_meet_esr(self):

        print("_" * 50)
        print("test_meet_esr")

        base = "./constraints/esr/"

        path1 = base + "model_esr_False1.dill"
        path2 = base + "model_esr_True1.dill"
        path3 = base + "model_esr_False2.dill"
        path4 = base + "model_esr_True2.dill"



        model1 = TAAMModel.load_model(path1)
        model2 = TAAMModel.load_model(path2)
        model3 = TAAMModel.load_model(path3)
        model4 = TAAMModel.load_model(path4)
        
        
        assert(not model1.meet_esr())
        assert(model2.meet_esr())
        assert(not model3.meet_esr())
        assert(model4.meet_esr())

        print("_" * 50)






    def test_meet_ensr(self):

        print("_" * 50)
        print("test_meet_ensr")

        base = "./constraints/ensr/"

        path1 = base + "model_ensr_False1.dill"
        path2 = base + "model_ensr_True1.dill"
        # path3 = base + "model_ensr_False2.dill"
        # path4 = base + "model_ensr_True2.dill"



        model1 = TAAMModel.load_model(path1)
        model2 = TAAMModel.load_model(path2)
        # model3 = TAAMModel.load_model(path3)
        # model4 = TAAMModel.load_model(path4)
        
        
        assert(not model1.meet_ensr())
        assert(model2.meet_ensr())
        # assert(not model3.meet_ensr())
        # assert(model4.meet_ensr())

        print("_" * 50)





    def test_meet_eos(self):

        print("_" * 50)
        print("test_meet_eos")

        base = "./constraints/eos/"

        path1 = base + "model_eos_False1.dill"
        path2 = base + "model_eos_True1.dill"
        path3 = base + "model_eos_False2.dill"
        path4 = base + "model_eos_True2.dill"



        model1 = TAAMModel.load_model(path1)
        model2 = TAAMModel.load_model(path2)
        model3 = TAAMModel.load_model(path3)
        model4 = TAAMModel.load_model(path4)
        
        
        assert(not model1.meet_eos())
        assert(model2.meet_eos())
        assert(not model3.meet_eos())
        assert(model4.meet_eos())

        print("_" * 50)


    def test_meet_das(self):

        print("_" * 50)
        print("test_meet_das")

        base = "./constraints/das/"

        path1 = base + "model_das_False1.dill"
        path2 = base + "model_das_True1.dill"
        path3 = base + "model_das_False2.dill"
        path4 = base + "model_das_True2.dill"
        path5 = base + "model_das_False3.dill"
        path6 = base + "model_das_True3.dill"


        model1 = TAAMModel.load_model(path1)
        model2 = TAAMModel.load_model(path2)
        model3 = TAAMModel.load_model(path3)
        model4 = TAAMModel.load_model(path4)
        model5 = TAAMModel.load_model(path5)
        model6 = TAAMModel.load_model(path6)       
        
        assert(not model1.meet_das())
        assert(model2.meet_das())
        assert(not model3.meet_das())
        assert(model4.meet_das())
        assert(not model5.meet_das())
        assert(model6.meet_das())
        print("_" * 50)


    def test_meet_nwci(self):

        print("_" * 50)
        print("test_meet_nwci")

        base = "./constraints/nwci/"

        path1 = base + "model_nwci_False1.dill"
        path2 = base + "model_nwci_True1.dill"
        path3 = base + "model_nwci_False2.dill"
        path4 = base + "model_nwci_True2.dill"
        path5 = base + "model_nwci_False3.dill"
        path6 = base + "model_nwci_True3.dill"


        model1 = TAAMModel.load_model(path1)
        model2 = TAAMModel.load_model(path2)
        model3 = TAAMModel.load_model(path3)
        model4 = TAAMModel.load_model(path4)
        model5 = TAAMModel.load_model(path5)
        model6 = TAAMModel.load_model(path6)       
        
        assert(not model1.meet_nwci())
        assert(model2.meet_nwci())
        assert(not model3.meet_nwci())
        assert(model4.meet_nwci())
        assert(not model5.meet_nwci())
        assert(model6.meet_nwci())
        print("_" * 50)


    def test_meet_faD(self):

        base = "./constraints/faD/"
        path1 = base + "faD_faW_model1.dill"
        path2 = base + "faD_faW_model2.dill"
        path3 = base + "faD_faW_model3.dill"
        path4 = base + "faD_faW_model4.dill"

        model1 = TAAMModel.load_model(path1)
        model2 = TAAMModel.load_model(path2)
        model3 = TAAMModel.load_model(path3)
        model4 = TAAMModel.load_model(path4)


        assert(model1.meet_faD())
        assert(not model2.meet_faD())
        assert(not model3.meet_faD())
        assert(model4.meet_faD())

        return

    def test_meet_faW(self):
        
        base = "./constraints/faW/"
        path1 = base + "faD_faW_model1.dill"
        path2 = base + "faD_faW_model2.dill"
        path3 = base + "faD_faW_model3.dill"
        path4 = base + "faD_faW_model4.dill"

        model1 = TAAMModel.load_model(path1)
        model2 = TAAMModel.load_model(path2)
        model3 = TAAMModel.load_model(path3)
        model4 = TAAMModel.load_model(path4)

        assert(model1.meet_faW())
        assert(not model2.meet_faW())
        assert(model3.meet_faW())
        assert(not model4.meet_faW())

        return






class TestBooleanAlgebra(object):

    def test_is_equivalent(self):
        
        from TAAMmodel import BooleanAlgebra as BA

        d2 = BA(2)
        a,b = d2.PROPVARS[0] , d2.PROPVARS[1]

        assert(BA.is_equivalent(a,a))
        assert(BA.is_equivalent(a | ~ a, S.true))
        assert(BA.is_equivalent(a & ~a,S.false))
        assert(BA.is_equivalent(a >> a,b >> b))

        assert(not BA.is_equivalent(a,b))

        return
    
    def test_is_tautology(self):

        from TAAMmodel import BooleanAlgebra as BA

        d2 = BA(2)
        a,b = tuple(d2.PROPVARS)

        assert(BA.is_tautology(a >> a))
        assert(BA.is_tautology(S.true))
        assert(BA.is_tautology((a >> a) & (b >> b)))

        assert(not BA.is_tautology(S.false))
        assert(not BA.is_tautology(a | b))

        return


        
        

        

    def test_draw(self):
        d1 = BooleanAlgebra(1)
        d2 = BooleanAlgebra(2)

        d1.visualize(title="BA1")
        d2.visualize(title="BA2")

        s = repr(d2)
    
    def test_is_included(self):

        d = BooleanAlgebra(2)
        a,b = d.PROPVARS[0],d.PROPVARS[1]

        assert(BooleanAlgebra.is_included([a,b],[S.true,a,b]))
        assert(BooleanAlgebra.is_included([a,b],[a,b]))
        assert(BooleanAlgebra.is_included([S.true], [a | ~ a]))
        assert(BooleanAlgebra.is_included([],[]))
        assert(BooleanAlgebra.is_included([a | ~ a,a,b],[b | ~ b,a,b,S.true]))


        assert(not BooleanAlgebra.is_included([a,b,a>>b],[S.true,a,b]))
        assert(not BooleanAlgebra.is_included([S.true], [a]))
        assert(not BooleanAlgebra.is_included([a,a,a],[b,b,b,b,S.true,S.false]))
        assert(not BooleanAlgebra.is_included([a >> a >> a],[b >> b >> b]))

    def test_is_boolean_algebra(self):
        from TAAMmodel import BooleanAlgebra as BA
        d1 = BA(1)
        a = d1.PROPVARS[0]

        assert(BA.is_boolean_algebra([S.true,S.false]))
        assert(BA.is_boolean_algebra([a,~a,a | ~ a,S.false]))
        assert(not BA.is_boolean_algebra([S.true]))
        assert(not BA.is_boolean_algebra([a,~a]))

        with pytest.raises(NotImplementedError):
            BA.is_boolean_algebra([])
        
        d2 = BooleanAlgebra(2)

        a,b = d2.PROPVARS[0],d2.PROPVARS[1]

        logic1 = a & b
        logic2 = ~a & ~ b

        assert(BA.is_boolean_algebra([S.false,b,~b,S.true ]))

        assert(not BA.is_boolean_algebra([S.false,S.true,a & b,a | b]))
        assert(not BA.is_boolean_algebra([S.false,S.true,logic1,logic2,~logic1,~logic2,logic1 & logic2]))
        assert(not BA.is_boolean_algebra([S.false,logic1,logic2,logic1 | logic2]))



    

    def test_calc_join(self):


        d = BooleanAlgebra(2)
        
        logic1 = d.gen_random_expr()
        logic2 = d.gen_random_expr()
        logic3 = d.gen_random_expr()
        res = BooleanAlgebra.calc_join([logic1,logic3,logic2])[0]
    
        assert(BooleanAlgebra.is_equivalent(res,logic1 | logic2 | logic3))
        assert(BooleanAlgebra.calc_join([]) == [])

    def test_calc_meet(self):
        d = BooleanAlgebra(2)

        
        logic1 = d.gen_random_expr()
        logic2 = d.gen_random_expr()
        logic3 = d.gen_random_expr()
        res = BooleanAlgebra.calc_meet([logic1,logic2,logic3])[0]

        assert(BooleanAlgebra.is_equivalent(res,logic1 & logic2 & logic3 & logic2))
        assert(BooleanAlgebra.calc_meet([]) == [])


    def test_uparrow(self):
        d = BooleanAlgebra(2)

        logic1 = d.PROPVARS[0]
        logic2 = d.PROPVARS[0] | d.PROPVARS[1]

        assert(d.in_uparrow(logic1,logic2))
        assert(not d.in_uparrow(logic2,logic1))
        assert(BooleanAlgebra.in_uparrow(logic1,logic1))
        assert(d.in_uparrow(S.false,logic1))
        assert(not d.in_uparrow(d.PROPVARS[0],d.PROPVARS[1]))

        assert(BooleanAlgebra.in_uparrow(d.PROPVARS[0] &  ~ d.PROPVARS[1],~ d.PROPVARS[0] | ~ d.PROPVARS[1]))
    
    def test_downarrow(self):

        d = BooleanAlgebra(2)

        logic1 = d.PROPVARS[0]
        logic2 = d.PROPVARS[0] | d.PROPVARS[1]

        assert(not d.in_downarrow(logic1,logic2))
        assert(d.in_downarrow(logic2,logic1))
        assert(BooleanAlgebra.in_downarrow(logic1,logic1))
        assert(not d.in_downarrow(S.false,logic1))
        assert(d.in_downarrow(S.true,logic1))
        assert(d.in_downarrow(S.false,S.false))
        assert(not BooleanAlgebra.in_downarrow(d.PROPVARS[0],d.PROPVARS[1]))
        assert(d.in_downarrow(~ d.PROPVARS[0] |  d.PROPVARS[0],d.PROPVARS[0]))

        assert(not d.in_downarrow(d.PROPVARS[0] &  ~ d.PROPVARS[1],~ d.PROPVARS[0] | ~ d.PROPVARS[1]))

    def test_private_minimal_representations(self):

        from TAAMmodel import BooleanAlgebra as BA

        d = BooleanAlgebra(2)
        a,b = d.PROPVARS[0],d.PROPVARS[1]

        def is_equal(test,ans) -> bool:

            for t in test:
                exists = False

                for a in ans:

                    exists |= BA.is_included(list(t),list(a)) and BA.is_included(list(a),list(t))
            
                if not exists:
                    return False

            for a in test:
                exists = False

                for t in ans:

                    exists |= BA.is_included(list(t),list(a)) and BA.is_included(list(a),list(t))
            
                if not exists:
                    return False


            return True
        
        logics  =  set([a & b,a & ~b ,~a & b, ~a & ~b])
        test = BA._calc_minimal_representations(logics)
        ans = [
            set([ a &  b,  a & ~b]),
            set([ a &  b, ~a &  b]),
            set([ a &  b, ~a & ~b]),
            set([ a & ~b, ~a &  b]),
            set([ a & ~b, ~a & ~b]),
            set([~a &  b, ~a & ~b]),
        ]

        assert(is_equal(test,ans))

        logics = set([a | b, a | ~b, a])
        test = BA._calc_minimal_representations(logics)
        ans = [
            set([ a ]),
            set([ a |  b, a | ~b]),
        ]

        assert(is_equal(test,ans))


        logics = set([a | b, a | ~b, a])
        test = BA._calc_minimal_representations(logics)
        ans = [
            set([ a |  b, a | ~b]),
            set([ a ]),
        ]

        assert(is_equal(test,ans))

        logics = set([a | b, a | ~b, a])
        test = BA._calc_minimal_representations(logics)
        ans = [
            set([  ]),
            set([ a |  b, a | ~b]),
        ]

        assert(not is_equal(test,ans))

        logics = set([a | b, a | ~b, a])
        test = BA._calc_minimal_representations(logics)
        ans = [
            set([ a |  b, a | ~b]),
        ]

        assert(not is_equal(test,ans))

        logics = set([a | b, a | ~b, a])
        test = BA._calc_minimal_representations(logics)
        ans = [
            set([a]),
        ]

        assert(not is_equal(test,ans))


        logics = set([S.true])
        test = BA._calc_minimal_representations(logics)
        ans = [
            set([S.true])
        ]

        assert(is_equal(test,ans))

        logics = set([S.true])
        test = BA._calc_minimal_representations(logics)
        ans = [
            set([S.true,S.false])
        ]

        assert(not is_equal(test,ans))


        logics = set([a,~a])
        test = BA._calc_minimal_representations(logics)
        ans = [
            set([~a,a])
        ]

        assert(is_equal(test,ans))

        logics = set([a,~a,S.false])
        test = BA._calc_minimal_representations(logics)
        ans = [
            set([S.false]),
            set([a,~a])
        ]

        assert(is_equal(test,ans))

        logics = set([a,~a,S.false])
        test = BA._calc_minimal_representations(logics)
        ans = [
            set([S.false])
        ]

        assert(not is_equal(test,ans))


        logics = set([a])
        test = BA._calc_minimal_representations(logics)
        ans = [
            set([b])
        ]

        assert(not is_equal(test,ans))


        logics = set([S.false,a])
        test = BA._calc_minimal_representations(logics)
        ans = [
            set([S.false])
        ]

        assert(is_equal(test,ans))


        logics = set([S.false,~a,a | ~ a])
        test = BA._calc_minimal_representations(logics)
        ans = [
            set([S.false])
        ]

        assert(is_equal(test,ans))

        logics = set([S.false,a])
        test = BA._calc_minimal_representations(logics)
        ans = [
            set([S.false])
        ]

        assert(is_equal(test,ans))

        logics = set([a | ~a])
        test = BA._calc_minimal_representations(logics)
        ans = [
            set([a | ~ a])
        ]

        assert(is_equal(test,ans))

        logics = set([S.false,~a,a | ~a])
        test = BA._calc_minimal_representations(logics)
        ans = [
            set([S.false])
        ]

        assert(is_equal(test,ans))
        return



    def test_prec1(self):

        from TAAMmodel import BooleanAlgebra as BA
        d = BooleanAlgebra(2)

        a,b = d.PROPVARS[0],d.PROPVARS[1]

        logics1 = set([])
        logics2 = set([])

        assert(d.prec1(logics1,logics2))

        logics1 = set([a])
        logics2 = set([b])

        assert(not d.prec1(logics1,logics2))

        logics1 = set([a,S.true])
        logics2 = set([a])

        assert(not d.prec1(logics1,logics2))
        assert(d.prec1(logics2,logics1))

        logics1 = set([])
        logics2 = set([a,b])

        assert(not d.prec1(logics1,logics2))
        assert(not d.prec1(logics2,logics1))

        logics1 = set([a,~b])
        logics2 = set([a,~b,S.true])

        assert(d.prec1(logics1,logics2))
        assert(not d.prec1(logics2,logics1))

        logics1 = set([a,~b])
        logics2 = set([a,~b,S.false])

        assert(not d.prec1(logics1,logics2))
        assert(not d.prec1(logics2,logics1))

        return




        