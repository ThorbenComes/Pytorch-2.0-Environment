# TODO add recurrent inputs and blocks made up of blocks

class block:
    """
    Class for determining earliest and latest computation time, modeling a single block of a model
    """
    def __init__(self, block_info, isSource=False, isTarget=False):
        self.predecessors = []  # list of id's of all direct predecessors
        self.block_info = block_info
        self.inputs = self.block_info[1]
        self.outputs = self.block_info[2]
        self.type = self.block_info[0]
        self.isSource = isSource
        self.isTarget = isTarget
        self.earliest = -1 if not self.isSource else 0  # if a block is a source, it is available at step one
        self.latest = -1
        self.id = -1  # defined as position within the block list plus the length of the source

    def add_subBlock(self, predecessor):
        self.predecessors.append(predecessor)


def build_model_graph(source, block_config, targets):
    # TODO add recurrent block inputs to source

    #### add source and targets to blocks

    blocks = [block(b) for b in block_config]
    blocks.extend([build_input_block(b) for b in source])
    blocks.extend([build_output_block(b) for b in targets])

    #### calculate execution order

    # assign ids to each block
    len_blocks = len(blocks)
    for i in range(len_blocks):
        blocks[i].id = i

    # set predecessor blocks
    set_predecessors(blocks)
    final_step = calculate_earliest_execution(blocks)
    # print(blocks)

    #### build graph
    graph = []
    for b in blocks:
        for p in b.predecessors:
            graph.append((p, b.id))

    return blocks, graph


def build_input_block(input_name):
    block_info = (input_name, [], [input_name], None)
    return block(block_info, isSource=True)


def build_output_block(output_name):
    block_info = (output_name, [output_name], [], None)
    return block(block_info, isTarget=True)


def find_predecessor(blocks, input):
    for b in blocks:
        if input in b.outputs:
            return b.id


def set_predecessors(blocks):
    # add block ids for each input
    for b in blocks:
        for input in b.inputs:
            b.predecessors.append(find_predecessor(blocks, input))

    # remove duplicates
    for b in blocks:
        b.predecessors = list(set(b.predecessors))


def calculate_earliest_execution(blocks):
    source = []
    not_calculated = [i for i in range(len(blocks))]
    # get initial source
    for b in blocks:
        if b.isSource:
            source.append(b.id)
            not_calculated.remove(b.id)

    step = 1
    while not_calculated:
        blocks_to_remove = []
        for b_id in not_calculated:
            b = blocks[b_id]
            if sum([pred in source for pred in b.predecessors]) == len(b.predecessors):
                b.earliest = step
                blocks_to_remove.append(b_id)
                for pred_block_id in b.predecessors:
                    blocks[pred_block_id].latest = step - 1

        # remove all calculated blocks
        for b_id in blocks_to_remove:
            not_calculated.remove(b_id)
            source.append(b_id)

        step += 1
    final_step = step - 1
    return final_step
