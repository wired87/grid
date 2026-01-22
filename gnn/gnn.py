"""

Update rocess
inject nodes
identify nodes e > 0
create G copy
"""
import json
import os

from flax import nnx

import jax
from jax import jit, vmap

from gnn.db_layer import DBLayer
from gnn.injector import InjectorLayer
from mod import Node
from utils import SHIFT_DIRS, create_runnable

try:
    from safetensors.numpy import save_file as safetensors_save
except Exception:
    safetensors_save = None

import jax.numpy as jnp


class GNN(
    nnx.Module
):
    # todo pattern applieance and converter -< data ar request,
    # build stabel update loop

    """
    zeit

    Universal Actor for all fields to process a batch of Fields same type)
    With a GPU

    Design the system so each method has its own preprocessor (CPU)
    and processor (GPU)

    receive struct:
    [
    soa
    ],

    [
    [runnables]
    [(soa indices for params)]
    }

    WF:
    Hold all data on gpu but calc diferentyl
    pattern must come from cpu

    todo cache results to index
    """

    def __init__(
            self,
            gpu,
            amount_nodes,
            time:int,
            feature_out_gnn,
            db_out_gnn,
            edge_db_to_method_variation_struct,
            iterator_skeleton,
            return_key_map,
            nodes_db=None,
            inj_pattern:list[int]=None,
            modules_len=0,
            gnn_data_struct:list=None,
            method_struct=None,
            def_out_db=None,
    ):
        if inj_pattern is not None:
             self.injection_pattern = inj_pattern
        else:
            injection_pattern = os.getenv("INJECTION_PATTERN")
            self.injection_pattern = json.loads(injection_pattern) if injection_pattern else []

        self.feature_out_gnn = feature_out_gnn

        self.global_utils = None

        self.gnn_data_struct=gnn_data_struct

        self.db_out_gnn=db_out_gnn
        self.iterator_skeleton = iterator_skeleton

        self.edge_db_to_method_variation_struct=edge_db_to_method_variation_struct
        self.len_params_per_method = iterator_skeleton["len_params_per_method"]


        self.index = 0
        self.return_key_map=return_key_map
        self.time = time
        self.nodes_db = nodes_db
        self.amount_nodes = amount_nodes

        self.world = []
        self.store = []
        self.module_map = []
        self.time=time
        self.model = []



        # Generate grid coordinates based on amount_nodes dimensionality
        self.schema_grid = [
            (i,i,i)
            for i in range(amount_nodes)
        ]

        self.len_params_per_methods = {}
        self.change_store = []

        self.modules_len=modules_len
        self.inj_pattern = inj_pattern

        self.inj_layer = InjectorLayer()

        self.db_layer = DBLayer(self.amount_nodes, gpu)

        # features jsut get etend naturally to the arr
        self.model_skeleton = jnp.array([])

        if method_struct is not None:
             self.method_struct = method_struct
        else:
            method_struct_env = os.getenv("METHOD_LAYER")
            self.method_struct = json.loads(method_struct_env) if method_struct_env else []

        if def_out_db is not None:
             self.def_out_db = jnp.array(def_out_db)
        else:
            def_out_db_env = os.getenv("METHOD_OUT_DB")
            self.def_out_db = jnp.array(json.loads(def_out_db_env)) if def_out_db_env else jnp.array([])

        print("Node initialized and build successfully")



    def main(self):
        self.build_gnn()
        self.simulate()
        model = self.build_model()
        serialized = self.serialize(model)
        self.serialized_model = serialized


    def simulate(self):
        try:
            for step in range(self.time):
                jax.debug.print(
                    "Sim step {s}/{n}",
                    s=step,
                    n=self.time
                )

                # apply injections inside db layer
                self.inj_layer.inject(
                    step=step,
                    db_layer=self.db_layer
                )
                
                # prepare gs
                self.db_layer.process_time_step()

                # perform calc step
                self.process_t_step()

        except Exception as e:
            jax.debug.print(f"Err simulate: {e}")
            raise



    def process_t_step(self):
        # includes actual params ready to calc
        # todo später: behalte features und patterns in den nodes -> transferier alles am ende in einem mal in X struktur

        # get params from DB layer and init calc
        all_results = self.calc_batch()

        # RESULT -> HISTORY DB todo: upsert directly to bq
        # todo: add param stacks on single eq layer
        vmap(
            self.scatter_results_to_db,
            in_axes=(0, 0)
        )(all_results, self.db_layer.method_to_db)



    def build_model(self):
        model = [
            # INJECTIOIN -> to get directly inj pattern (todo switch throguh db mapping)
            # DB SCHEMA
            self.db_layer.db_pattern,

            # EDGE DB -> METHOD
            self.method_struct,

            # FEATURES
            self.model_skeleton,

            # FEATURE DB
            self.def_out_db
        ]
        return model


    def calc_batch(self):
        # calc all methods and apply result to new g

        all_results = []

        for mod_idx, amount_equations in enumerate(
                self.iterator_skeleton["modules"]
        ):

            #
            for eq_idx in range(amount_equations):
                node:Node = self.get_equation(
                    mod_idx=mod_idx,
                    eq_idx=eq_idx,
                )

                # extract values
                variations = self.extract_field_variations(
                    mod_idx,
                    eq_idx,
                )


                params:list[jnp.array] = self.get_param_items()

                # calc single equation
                features, results = node(
                    self.db_layer.old_g,
                    param_struct,
                )

                # append all features to final struct
                self.model_skeleton = jnp.concatenate(
                    [
                        self.model_skeleton,
                        features.reshape(-1)
                    ]
                )

        return all_results



    def sort_features(self, eq_idx, all_features):
        len_eq_variations = self.iterator_skeleton["eq_variations"][eq_idx]
        start_idx = jnp.sum(self.iterator_skeleton["eq_variations"])[:eq_idx]

        indices = jnp.array(start_idx+i for i in range(len_eq_variations))
        # FEATURE -> MODEL
        self.model_skeleton.at[
            tuple(indices.T)
        ].add(all_features)




        """
            def gnn_skeleton(self, feature_schema):
        # feature
        jax.debug.print("gnn_skeleton...")

        # SET EMPTY STRUCTURE OF MODEL
        model_skeleton = jnp.array([
            jnp.array([])
            for _ in range(
                jnp.sum(self.iterator_skeleton["eq_variations"]))
        ])

        jax.debug.print("gnn_skeleton... done")
        return model_skeleton
        """




    def extract_field_variations(self, mod_idx, eq_idx):





    def extract_field_variations(self, mod_idx, eq_idx):
        """
        Extrahiert alle Variationen für ein spezifisches Feld/Gleichung.
        Navigiert durch DB und AXIS und skaliert Parameter bei axis == 0 auf amount_nodes.
        """
        jax.debug.print("📊 Extracting Variations for Mod {m} Eq {e}", m=mod_idx, e=eq_idx)

        # sum amount equations till current + all their fields
        variation_start = sum(self.iterator_skeleton["fields"][:mod_idx]) + sum(self.iterator_skeleton["module"][:mod_idx])

        # get eq len
        variation_len = self.iterator_skeleton["fields"][mod_idx]



        variation_end = variation_start + variation_len

        # get variation block
        variations = self.edge_db_to_method_variation_struct[variation_start:variation_end]
        return variations







        # 1. Bestimme den Startpunkt im db_params Mapping für dieses Modul/Eq
        # Wir nutzen die Metadaten aus dem iterator_skeleton
        num_eqs_before = sum(self.iterator_skeleton["modules"][:mod_idx])
        global_eq_idx = num_eqs_before + eq_idx

        # Offsets berechnen: Wo liegen die Indizes für diese Gleichung im db_params?
        # (Annahme: db_params ist flach gespeichert: [Eq1_Field1_Vars, Eq1_Field2_Vars, ...])
        num_fields = iterator_skeleton["fields"][mod_idx]

        # Wir loopen über die Felder dieses Moduls
        all_field_results = []

        for f_idx in range(num_fields):
            # Index für field_variations finden
            # (Jedes Feld hat eine eigene Anzahl an Variationen)
            var_count_idx = (sum(iterator_skeleton["fields"][:mod_idx]) * num_eqs_before) + (
                        f_idx * num_eqs_before) + eq_idx
            num_vars = iterator_skeleton["field_variations"][var_count_idx]
            params_per_eq = iterator_skeleton["method_param"][global_eq_idx]

            # 2. Extrahiere die db_indices für alle Variationen dieses Feldes
            # Wir berechnen den Slice im flachen iterator_skeleton["db_params"]
            start_mapping = sum(iterator_skeleton["field_variations"][:var_count_idx]) * params_per_eq
            end_mapping = start_mapping + (num_vars * params_per_eq)

            # Das sind die "Zettel" mit den Adressen im flat_db
            mapping_indices = jnp.array(iterator_skeleton["db_params"][start_mapping:end_mapping]).reshape(
                (num_vars, params_per_eq))

            # 3. Innere Funktion für die Extraktion (wird per vmap über Variationen ausgeführt)
            def process_variation(var_indices):
                variation_data = []
                for db_idx in var_indices:
                    # Prüfe die Achse im flachen AXIS-Buffer
                    axis_val = flat_axis[db_idx]

                    # Bedingung: axis == 0 -> Gesamten Node-Block extrahieren
                    # JAX-konforme Umsetzung der if-Logik via select oder dynamic_slice
                    def get_node_block():
                        return jax.lax.dynamic_slice_in_dim(flat_db, db_idx, amount_nodes)

                    def get_single_val():
                        # Falls axis != 0, nimm nur den einen Wert und padde ihn (oder broadcast)
                        val = flat_db[db_idx]
                        return jnp.broadcast_to(val, (amount_nodes,))

                    # Die GPU entscheidet hier maskiert, welcher Pfad genommen wird
                    node_param_vector = jax.lax.cond(axis_val == 0, get_node_block, get_single_val)
                    variation_data.append(node_param_vector)

                return jnp.stack(variation_data)  # Form: (Params_pro_Eq, Amount_Nodes)

            # 4. Vektorisierte Ausführung über alle Variationen des Feldes
            # Das macht die 100.000 Nodes und n-Variationen extrem schnell
            extracted_vars = jax.vmap(process_variation)(mapping_indices)
            all_field_results.append(extracted_vars)

        jax.debug.print("✅ Variations extracted.")
        return all_field_results





    def get_equation(self, mod_idx, eq_idx):
        # retirn index from methdod sruct to gather either Node or str eq
        def_idx = jnp.sum(self.iterator_skeleton["modules"][mod_idx:]) + eq_idx
        return def_idx


    def get_methods_axis_def(self, mod_idx):
        """
        module eq list -> sum till current ->
        """
        # Start ist die Summe der Elemente vor dem Index
        start = jnp.sum(self.iterator_skeleton["modules"][mod_idx:])

        # Ende ist die Summe der Elemente bis einschließlich des Index
        end = start + self.iterator_skeleton["modules"][mod_idx+1]

        return self.iterator_skeleton["def_axis"][start:end]



    def build_gnn(self):
        for mod_idx, amount_equations in enumerate(
                self.iterator_skeleton["modules"]):

            for eq_idx in range(amount_equations):
                def_idx = self.get_equation(mod_idx, eq_idx)
                eq_axis_def = self.get_methods_axis_def(mod_idx)

                node = Node(
                    runnable=create_runnable(
                        eq_code=self.method_struct[def_idx],
                    ),
                    in_axes_def=eq_axis_def,
                    method_id=eq_idx,
                    mod_idx=mod_idx,
                )
                # replace mehod str with py class
                self.method_struct[def_idx] = node



    @jit
    def scatter_results_to_db(self, results, db_start_idx):
        """
        Schreibt Berechnungs-Resultate zurück in den flachen DB-Buffer.

        flat_db: Der aktuelle 1D Parameter-Buffer.
        flat_axis: Der 1D Buffer mit den Achsen-Definitionen (0 oder None).
        results: Das JNP-Array der neuen Werte (Länge: 1 oder amount_nodes).
        db_start_idx: Der Start-Index im flat_db, wo die Ersetzung beginnt.
        amount_nodes: Die Anzahl der räumlichen Knoten.
        """

        # 1. Bestimme die Achsen-Regel am Startpunkt
        axis_rule = self.db_layer.axis[db_start_idx]

        def update_field_block(db, res):
            # Fall: axis == 0 -> Wir ersetzen einen Block der Länge n
            # Wir nutzen dynamic_update_slice für maximale GPU-Performance
            return jax.lax.dynamic_update_slice_in_dim(
                db,
                res.reshape(-1),  # Sicherstellen, dass es 1D ist
                db_start_idx,
                axis=0
            )

        def update_single_value(db, res):
            # Fall: axis == None -> Wir ersetzen nur einen einzelnen Wert
            # Falls das Resultat ein Vektor ist (z.B. durch Reduktion), nehmen wir den Durchschnitt
            # oder das erste Element, je nach physikalischer Logik.
            val = res[0] if res.ndim > 0 else res
            return db.at[db_start_idx].set(val)

        # 2. Bedingte Ausführung auf der GPU
        # jax.lax.cond vermeidet Python-Side-Effects und läuft komplett auf dem Core
        self.db_layer.new_g = jax.lax.cond(
            axis_rule == 0,
            update_field_block,
            update_single_value,
            self.db_layer.new_g,
            results
        )






    def serialize(self, data):
        import flax.serialization
        # Wandelt den State in einen Byte-String um
        binary_data = flax.serialization.to_bytes(data)
        return binary_data


    def set_shift(self, start_pos:list[tuple]=None):
        """
        Calculates neighboring node indices based on SHIFT_DIRS using pure-Python coordinate addition 
        to avoid JAX type issues during initialization.
        """
        if start_pos is None:
            # If no starting positions are provided, we initialize for all nodes in the grid
            start_pos = self.schema_grid
            
        next_index_map = []
        for pos in start_pos:
            # SHIFT_DIRS[0] + SHIFT_DIRS[1] combines positive and negative directions
            for d in (SHIFT_DIRS[0] + SHIFT_DIRS[1]):
                # Pure python addition: tuple zip sum
                neighbor_pos = tuple(a + b for a, b in zip(pos, d))
                
                if neighbor_pos in self.schema_grid:
                    next_index_map.append(
                        self.schema_grid.index(neighbor_pos)
                    )
            # include the node itself
            next_index_map.append(
                self.schema_grid.index(pos)
            )
        return next_index_map


    def get_index(self, pos: tuple) -> int:
        """Returns the integer index of a grid position."""
        try:
            return self.schema_grid.index(pos)
        except ValueError:
            # Fallback or error handling
            return -1


    def set_next_nodes(self, pos_module_field_node):
        for module in pos_module_field_node:
            for field_pos_list in module:
                field_pos_list = self.set_shift(field_pos_list)
        return pos_module_field_node


    def get_sdr_rcvr(self):
        receiver = []
        sender = []
        for i, item in enumerate(range(len(self.store))):
            for j, eitem in enumerate(self.edges):
                sender.extend([i for _ in range(len(self.edges))])
                receiver.extend(eitem)
        jax.debug.print("set direct interactions finsihed")
        return sender, receiver

"""


    def get_params(self, mod_idx, eq_idx):

        eq_start_idx = sum(self.iterator_skeleton["modules"][:mod_idx])
        eq_end_idx = eq_start_idx + self.iterator_skeleton["modules"][mod_idx]

        # 2. Bestimme den Parameter-Bereich im flachen Array (flat_params)
        # Dazu summieren wir die 'len_params_per_method' Werte (Anzahl Parameter pro Gleichung)
        # Start-Punkt: Summe aller Parameter-Anzahlen vor dem ersten Index dieses Moduls
        param_start_offset = sum(self.len_params_per_method[:eq_start_idx])

        # End-Punkt: Start-Punkt + Summe der Parameter-Anzahlen innerhalb dieses Moduls
        params_in_module = sum(self.len_params_per_method[eq_start_idx:eq_end_idx])
        param_end_offset = param_start_offset + params_in_module

        # 3. Extraktion params
        module_slice = flat_params[param_start_offset:param_end_offset]



    def sort_features_to_fields(self, mod_idx, flat_equation_results):
       
        Sortiert die flachen Ergebnisse einer Gleichungs-Berechnung
        zurück in die field_variations Struktur.

        flat_equation_results: jnp.array (1D) - Alle berechneten Werte eines Moduls
        mod_idx: int - Welches Modul wurde berechnet
        
        jax.debug.print("Sorting results for mod {m}...", m=mod_idx)

        # Wie viele Gleichungen hat dieses Modul?
        num_eqs = self.iterator_skeleton["modules"][mod_idx]

        # Wie viele Felder hängen an diesem Modul?
        num_fields = self.iterator_skeleton["fields"][mod_idx]

        # Start- und End-Index im globalen field_variations Skeleton finden
        eq_start_glob = sum(self.iterator_skeleton["modules"][:mod_idx])
        # Die Anzahl der Einträge in field_variations pro Modul ist: num_eqs * num_fields
        var_start_idx = sum(self.iterator_skeleton["modules"][:mod_idx]) * num_fields
        var_end_idx = var_start_idx + (num_eqs * num_fields)

        # Das sind die Längen jeder Variationseinheit
        variation_lengths = jnp.array(self.iterator_skeleton["field_variations"][var_start_idx:var_end_idx])

        # Wir brauchen die exakten Stellen, an denen das 1D Array zerschnitten wird
        split_indices = jnp.cumsum(variation_lengths)[:-1]

        # Dies zerlegt das 1D Array in eine Liste von jnp.arrays
        sorted_fields = jnp.split(flat_equation_results, split_indices)

        structured_res = [
            sorted_fields[i * num_fields: (i + 1) * num_fields]
            for i in range(num_eqs)
        ]

        jax.debug.print("Sorting done. Created {e} equations with {f} fields each.",
                        e=len(structured_res), f=num_fields)

        return structured_res


   def grab_db_to_method(self, mod_idx, eq_idx):
        # return list[field[interactions_fro_equation]]
        jax.debug.print("grab_db_to_method...")

        # 1. Anzahl der Felder (muss statisch oder über self.def_out_db bekannt sein)
        len_fields = len(self.db_out_gnn[mod_idx])

        raw_indices = [
            self.db_out_gnn[mod_idx][i][eq_idx]
            for i in range(len(self.db_out_gnn[mod_idx]))
        ]

        for i in raw_indices:
            print("i", i)

        # 2. Umwandlung in ein JAX-Array, damit .T funktioniert
        # Form: (Anzahl_Felder, Dimensionen_der_Nodes)
        indice_map = [
            jnp.array(_indices)
            for _indices in raw_indices
        ]

        product = []
        # RESULTS
        for _indice in indice_map:
            params = self.db_layer.nodes.at[
                tuple(_indice.T)
            ].get()
            product.append(params)

        print("grab_db_to_method... done", len(product))
        return product






    def db_to_method(self, mod_idx, eq_idx, old_g):
        jax.debug.print("db_to_method für Modul {m}, Eq {e}...", m=mod_idx, e=eq_idx)


        # 1. Bestimme den globalen Index der Gleichung
        # Summe der Gleichungen aller vorherigen Module + aktueller lokaler eq_idx
        global_eq_idx = jnp.sum(
            jnp.array(
                self.iterator_skeleton["modules"][:mod_idx]
            )
        ) + eq_idx

        # get start from
        start_offset = jnp.sum(
            jnp.array(self.len_params_per_method[:global_eq_idx])
        )

        # get len params fro eq
        param_len = self.len_params_per_method[global_eq_idx]

        # 4. Extraktion via Dynamic Slice (GPU optimiert)
        # Wir nutzen dynamic_slice, um innerhalb von JIT stabil zu bleiben
        params = jax.lax.dynamic_slice_in_dim(old_g, start_offset, param_len)

        jax.debug.print("db_to_method... done. Extrahierte {l} Parameter", l=param_len)
        return params, (start_offset, start_offset + param_len)



"""