import random
from typing import Optional, Tuple
from kuimaze2 import MDPProblem
from kuimaze2.typing import VTable, QTable, Policy
from kuimaze2.map_image import map_from_image


class MDPAgent:
    """ Common base for MDP agents (both Value Iteration and Policy Iteration ) """

    def __init__(self, env: MDPProblem, gamma: float = 0.9, epsilon: float = 0.001):
        """ Initialize the agent with the environment, discount factor and epsilon """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

    def render(
        self,
        values: Optional[VTable] = None,
        qvalues: Optional[QTable] = None,
        policy: Optional[Policy] = None,
        **kwargs,
    ):
        """ Display the current state of the environment, including values, Q-values and policy """
        if values is None:
            values = {state: -1 + 2 * random.random() for state in self.env.get_states()}
        value_texts = {state: f"{value:.2f}" for state, value in values.items()}
        
        if qvalues is None:
            qvalues = {
                state: {action: -1 + 2 * random.random() for action in self.env.get_actions(state)}
                for state in self.env.get_states()
                
            }
        qvalue_texts = {
            (state, action): f"{q:.2f}"
            for state, actions in qvalues.items()
            for action, q in actions.items()
        }

        policy_texts = {}
        if policy is not None:
            policy_texts = {state: f"{policy[state]}" for state in self.env.get_states()}

        self.env.render(
            square_colors=values,
            square_texts=value_texts,
            triangle_colors=qvalues,
            triangle_texts=qvalue_texts,
            middle_texts=policy_texts,
            **kwargs,
        )

class ValueIterationAgent(MDPAgent):
    """ Agent that finds the optimal policy using Value Iteration """

    def init_values(self) -> VTable:
        """ Starting values for the value iteration """
        return {state: 0.0 for state in self.env.get_states()}

    def find_policy(self) -> Policy:
        """ Find the optimal policy using Value Iteration """
        values = self.init_values()
        
        # Iteratively update state values until convergence
        while True:
            delta = 0
            new_values = {}

            for state in self.env.get_states():
                actions = self.env.get_actions(state)
                if not actions:
                    new_values[state] = 0.0 # Terminal state
                    continue

                max_value = float("-inf")
                r = self.env.get_reward(state)
                for action in actions:
                    # Get the immediate reward once the action is taken
                    q_value = r
                    for next_state, prob in self.env.get_next_states_and_probs(state, action):
                        q_value += prob * (self.gamma * values.get(next_state, 0.0))
                    max_value = max(max_value, q_value)
                new_values[state] = max_value
                delta = max(delta, abs(values[state] - new_values[state]))
            values = new_values
            if delta < self.epsilon:
                break
        # Derive the policy by choosing the best action for each state
        policy = {}
        for state in self.env.get_states():
            actions = self.env.get_actions(state)
            if not actions:
                continue

            best_action = None
            best_q_value = float("-inf")
            for action in actions:
                
                q_value = self.env.get_reward(state)
                for next_state, prob in self.env.get_next_states_and_probs(state, action):
                    q_value += prob * (self.gamma * values.get(next_state, 0.0))

                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action

            policy[state] = best_action

        return policy
                
class PolicyIterationAgent(MDPAgent):
    """ Agent finds the optimal policy using Policy Iteration """

    def init_policy(self) -> Policy:
        """ Create a random policy to start the policy iteration """
        return {
            state: random.choice(self.env.get_actions(state)) if self.env.get_actions(state) else None 
            for state in self.env.get_states()
        }

    def evaluate_policy(self, policy: Policy) -> VTable:
        """Iteratively evaluate the given policy until values converge."""
        values = {state: 0.0 for state in self.env.get_states()}
        while True:
            delta = 0.0
            new_values = {}

            for state in self.env.get_states():
                action = policy.get(state)
                if action is None:
                    new_values[state] = 0.0
                    continue
                
                r = self.env.get_reward(state)
                q_value = r
                for next_state, prob in self.env.get_next_states_and_probs(state, action):
                    q_value += prob * (self.gamma * values.get(next_state, 0.0))

                new_values[state] = q_value
                delta = max(delta, abs(values[state] - new_values[state]))
            
            values = new_values

            if delta < self.epsilon:
                break

        return values
    
    def policy_improvement(self, values: VTable, policy: Policy) -> Tuple[Policy, bool]:
        """
        Improve the policy based on the given values.
        Returns the new policy and whether it is stable.
        """
        policy_stable = True
        for state in self.env.get_states():
            actions = self.env.get_actions(state)
            if not actions:
                continue

            old_action = policy[state]
            best_action = None
            best_q_value = float("-inf")
            
            for action in actions:
                r = self.env.get_reward(state)
                q_value = r
                
                for next_state, prob in self.env.get_next_states_and_probs(state, action):
                    q_value += prob * (self.gamma * values.get(next_state, 0.0))
                
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            
            policy[state] = best_action

            if best_action != old_action:
                policy_stable = False

        return policy, policy_stable
    
    def find_policy(self) -> Policy:
        """ Run policy iteration until policy no longer changes """
        policy = self.init_policy()
        while True:
            values = self.evaluate_policy(policy)
            policy, policy_stable = self.policy_improvement(values, policy)
            
            if policy_stable:
                break

        return policy

if __name__ == "__main__":
    from kuimaze2 import Map
    from kuimaze2.map_image import map_from_image

    # A simple maze representation as a maze
    MAP = """
    ...G
    .#.D
    S...
    """
    map = Map.from_string(MAP)
    # Alternatively, load a map from an image:
    # map = map_from_image("./maps/normal/normal3.png")
    
    # Create the MDP problem environment
    env = MDPProblem(
        map,
        action_probs=dict(forward=0.8, left=0.1, right=0.1, backward=0.0),
        graphics=True,
    )

    print(env.get_states())
    # agent = ValueIterationAgent(env, gamma=0.9, epsilon=0.001)
    agent = PolicyIterationAgent(env, gamma=0.9, epsilon=0.001)
    policy = agent.find_policy()
    print("Policy found:", policy)
    agent.render(policy=policy, wait=True)


