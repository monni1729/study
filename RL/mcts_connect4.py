import time
import numpy as np


class State:
    def __init__(self, state, turn='Black'):
        self.state = state
        self.turn = turn  # Black -> 0, White -> 1

    def get_legal_actions(self):
        if self.turn == 'Black':
            turn_num = 0
        else:
            turn_num = 1

        actions_list = []
        for i in range(7):
            stack = np.sum(self.state[:, :, i])
            if stack < 6:
                actions_list.append([turn_num, int(5 - stack), i])

        # print(actions_list)
        return actions_list

    def get_result(self):
        for j in range(6):
            for k in range(4):
                if np.sum(self.state[0, j, k:k + 4]) == 4:
                    return 'Black'
        for j in range(3):
            for k in range(7):
                if np.sum(self.state[0, j:j + 4, k]) == 4:
                    return 'Black'
        for j in range(3):
            for k in range(4):
                sum_4 = 0
                for l in range(4):
                    sum_4 += self.state[0, j + l, k + l]
                if sum_4 == 4:
                    return 'Black'
            for k in range(3, 7):
                sum_4 = 0
                for l in range(4):
                    sum_4 = self.state[0, j + l, k - l]
                if sum_4 == 4:
                    return 'Black'

        for j in range(6):
            for k in range(4):
                if np.sum(self.state[1, j, k:k + 4]) == 4:
                    return 'White'
        for j in range(3):
            for k in range(7):
                if np.sum(self.state[1, j:j + 4, k]) == 4:
                    return 'White'
        for j in range(3):
            for k in range(4):
                sum_4 = 0
                for l in range(4):
                    sum_4 += self.state[1, j + l, k + l]
                if sum_4 == 4:
                    return 'White'
            for k in range(3, 7):
                sum_4 = 0
                for l in range(4):
                    sum_4 = self.state[1, j + l, k - l]
                if sum_4 == 4:
                    return 'White'

        if np.sum(self.state) == 42:
            return 'Draw'
        else:
            return 'Going_on'

    def action(self, action):
        next_state = np.copy(self.state)
        next_state[action[0], action[1], action[2]] = 1

        if action[0] == 0:
            next_turn = 'White'
        else:
            next_turn = 'Black'

        return State(next_state, next_turn)

    def is_over(self):
        if self.get_result() == 'Going_on':
            return False
        else:
            return True


class Node:
    def __init__(self, state, parent):
        self.visits = 0
        self.reward = 0
        self.state = state
        self.parent = parent

        self.untried_actions = self.state.get_legal_actions()
        self.children = []
        self.wins = 0
        self.draws = 0
        self.loses = 0

    def back_pro(self, result):
        self.visits += 1
        if result == 'Black':
            if self.state.turn == 'Black':
                self.wins += 1
            else:
                self.loses += 1
        elif result == 'White':
            if self.state.turn == 'Black':
                self.loses += 1
            else:
                self.wins += 1
            pass
        elif result == 'Draw':
            self.draws += 1

        if self.parent:
            self.parent.back_pro(result)

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.action(action)
        child_node = Node(next_state, parent=self)

        self.children.append(child_node)
        return child_node

    def rollout(self):
        current_rollout_state = self.state
        while not current_rollout_state.is_over():
            legal_actions = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(legal_actions)
            current_rollout_state = current_rollout_state.action(action)

        return current_rollout_state.get_result()

    @staticmethod
    def rollout_policy(legal_actions):
        return legal_actions[np.random.randint(len(legal_actions))]

    def best_child(self, c_param=1.4):
        children_weights = []
        for child in self.children:
            weight = ((child.wins - child.loses) / child.visits) + (
                    c_param * np.sqrt((2 * np.log(self.visits) / child.visits)))
            children_weights.append(weight)
        return self.children[np.argmax(children_weights)]

    def is_terminal_node(self):
        if self.state.get_result() == 'Going_on':
            return False
        else:
            return True

    def is_fully_expanded(self):
        if len(self.untried_actions) == 0:
            return True
        else:
            return False


class Tree:
    def __init__(self, node):
        self.root = node

    def best_action(self, simulations):
        for _ in range(simulations):
            temp_node = self.tree_policy()
            reward = temp_node.rollout()
            temp_node.back_pro(reward)

        return self.root.best_child(c_param=0.)

    def tree_policy(self):
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node


if __name__ == '__main__':
    t_1 = time.time()

    init_state = State(state=np.zeros((2, 6, 7)), turn='Black')
    root = Node(init_state, parent=None)
    tree = Tree(root)

    for _ in range(1):
        tree.best_action(500)
        print("check")
        print('wins : {}'.format(root.wins))
        print('draws : {}'.format(root.draws))
        print('loses : {}'.format(root.loses))

    print("processing time : ", time.time() - t_1)
