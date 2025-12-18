import jax.numpy as jnp

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


        self.soa = {}


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


