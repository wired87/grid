"""
jnp.where(cond, a, b): elementweise If-Else.

total_fields_idx > 0: Bedingung pro Eintrag.

True-Fall:
SCALED_PARAMS_CUMSUM[total_fields_idx - 1]
→ nimmt die kumulative Summe bis zum vorherigen Feld ⇒ Startindex.

False-Fall (total_fields_idx == 0):
0 ⇒ erstes Feld startet bei Index 0.
"""