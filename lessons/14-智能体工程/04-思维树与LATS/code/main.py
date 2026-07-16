# 思维树 ToT 实现


class ThoughtNode:
    """思维树节点。"""
    def __init__(self, thought, parent=None):
        self.thought = thought
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0

    def expand(self, child_thoughts):
        for t in child_thoughts:
            self.children.append(ThoughtNode(t, parent=self))
        return self.children


class ThoughtTree:
    def __init__(self):
        self.root = None

    def build(self, initial_thought):
        self.root = ThoughtNode(initial_thought)
        return self.root

    def select_best_path(self):
        node = self.root
        path = [node.thought]
        while node.children:
            node = max(node.children, key=lambda n: n.value)
            path.append(node.thought)
        return path


def tot_search(problem, llm_fn, evaluator_fn, n_branches=3, depth=3):
    """ToT BFS 搜索。"""
    tree = ThoughtTree()
    tree.build(f"问题: {problem}")
    current_leaves = [tree.root]

    for d in range(depth):
        all_children = []
        for leaf in current_leaves:
            candidates = llm_fn(leaf.thought, n_candidates=n_branches)
            children = leaf.expand(candidates)
            for child in children:
                child.value = evaluator_fn(child.thought)
            all_children.extend(children)
        current_leaves = sorted(all_children, key=lambda n: -n.value)[:n_branches]

    return tree.select_best_path()


if __name__ == "__main__":
    print("思维树 ToT 演示\n")

    def mock_llm(thought, n_candidates=3):
        return [f"思考步骤 {i+1}（基于：{thought[:20]}...）" for i in range(n_candidates)]

    def mock_evaluator(thought):
        return random.random()

    path = tot_search("计算 24 点: 4,5,6,7", mock_llm, mock_evaluator)
    print("最佳推理路径:")
    for i, step in enumerate(path):
        print(f"  {i}: {step[:60]}")
