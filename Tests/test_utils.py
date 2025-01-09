"""tests for utils"""
import utils.hydra_utils


def test_prettyPrint():
    print("The lines of the string should have as many tabs as their name suggests")
    x = {"no_tab_1": {"One_tab_1": 2}, "no_tab_2": 26, "no_tab_3": {"One_tab_2": {"Three_tab": 29}}}
    utils.hydra_utils.prettyPrint(x)


test_prettyPrint()
