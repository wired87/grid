import jax.numpy as jnp
from jax import jit

from app_utils import ENVC
from data import GAUGE_FIELDS, QUARKS, FERM_PARAMS

from sm.fermion.ferm_creator import FermCreator
from sm.higgs.higgs_creator import HiggsCreator
from sm.gauge.g_creator import GaugeCreator

class CreateWorld:

    """

    HANDLES THE CREATION OF A CLUSTER
    Creator for bare empty sism World
    Start/Change ->
    Adapt Config ->
    Model Changes ->
    Results Upload Graph ->
    Render.

    """

    def __init__(
            self,
            amount_nodes
    ):

        # Creator classes
        self.g_creator = GaugeCreator(
            g_utils=None,
        )

        self.ferm_creator = FermCreator(
            g=None,
        )

        self.higgs_creator = HiggsCreator(
            g=None,
        )

        self.amount_nodes = amount_nodes
        self.gv = list(GAUGE_FIELDS.values())
        self.quarks = list(QUARKS.values())
        self.ferms = list(FERM_PARAMS.values())

        self.soa = {}


    def create_world(self, ntype):
        # todo
        dim=3

        self.world = jnp.meshgrid(
            jnp.array(
                self.get_point(
                    ntype,
                    jnp.int64(
                        ENVC["d"] * i
                    )
                )
                for i in range(self.world_cfg["amount_nodes"])
            )for _ in range(dim)
        )





    def get_point(self, ntype, pos):
        # todo optimize ram used, by loop adapted pos over zero point field
        # todo integrat just operator interaction rules, so in eac iter you calc param1 -> param2 for each field. follow the pathway (e.g. param1+param2*param3and4)
        """
        # save as bytes
        int(i | pos) # or merge str(i)+str(pos)
        for i in range(len(ALL_SUBS))
        """

        field = []
        # Sammle PHI-Daten
        field.append(
            self.higgs_creator.higgs_params_batch(
                True
            )
        )

        """
        format:
        [
        [field_index:int, [items map:int]]
        ]
        """

        index_map = [
            [i, [j for j in range(len(list(v)))]]
            for i, v in enumerate(field)
        ]

        print("index_map", index_map)
        return [pos, jnp.array(jnp.array(field))]


