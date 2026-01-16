import jax
import jax.numpy as jnp


class GNNQueryManager:
    def __init__(self, model_arsenal):
        """
        model_arsenal: Ein Dictionary mit deinen geladenen Modellen
        z.B. {"gcn": gcn_model, "temporal": tgnn_model}
        """
        self.arsenal = model_arsenal

    def _preprocess_list(self, raw_list, max_features=15):
        """Wandelt deine verschachtelte Liste in einen sauberen JAX-Tensor."""

        # Extrahiere nur numerische Werte, ersetze null durch 0.0
        def clean(val):
            if val is None or isinstance(val, str): return 0.0
            if isinstance(val, list): return sum(val)  # Beispielhafte Reduktion
            return float(val)

        # Flachklopfen der Struktur auf eine feste Feature-Größe
        processed = []
        for time_step in raw_list:
            step_data = []
            for field in time_step:
                # Extrahiere die Zahlen aus deinem Feld-Eintrag
                flat_field = [clean(x) for x in field[0]]  # Nimmt den ersten Teil des Eintrags
                # Padding auf max_features
                padded = flat_field[:max_features] + [0.0] * (max_features - len(flat_field))
                step_data.append(padded)
            processed.append(step_data)

        return jnp.array(processed)

    def query(self, algorithm_name, input_list, adj_matrix):
        """Die Hauptmethode für deine Abfrage."""
        # 1. Daten vorbereiten
        x = self._preprocess_list(input_list)  # Shape: (Zeit, Nodes, Features)

        # 2. Modell aus Arsenal wählen
        model = self.arsenal.get(algorithm_name)
        if model is None:
            raise ValueError(f"Algorithmus {algorithm_name} nicht im Arsenal!")

        # 3. Ausführung basierend auf Typ
        if algorithm_name == "gcn":
            # GCN schaut sich meist nur den letzten Zeitschritt an
            return model(x[-1], adj_matrix)

        elif algorithm_name == "temporal":
            # Temporal GNN iteriert über die gesamte Zeit-Achse
            # Nutzt jax.lax.scan für effiziente Iteration in JAX
            def scan_fn(carry, x_t):
                h_prev = carry
                h_next = model(x_t, adj_matrix, h_prev)
                return h_next, h_next

            init_h = jnp.zeros((x.shape[1], model.embedding_dim))
            final_h, history = jax.lax.scan(scan_fn, init_h, x)
            return history  # Gibt die Entwicklung über alle Zeitschritte zurück