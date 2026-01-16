from flax import nnx


class FieldNode(
    nnx.Module
):

    def __init__(
            self,
            data,
            input_patterns,
            output_patterns,
            mod_id,
    ):
        self.data = data
        self.mod_id = mod_id
        self.input_patterns = input_patterns
        self.output_patterns = output_patterns
        self.graph=None

    def set_graph(self, graph):
        self.graph = graph

    def get_inputs(self, midx):
        input_pattern = self.input_patterns[midx]
        output_patterns = self.output_patterns[midx]
        return input_pattern, output_patterns


    def get_(self, index):
        pass