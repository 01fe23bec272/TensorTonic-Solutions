def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    new_values = []

    num_states = len(values)

    for s in range(num_states):
        best = float("-inf")

        for a in range(len(transitions[s])):
            q = rewards[s][a]

            future = 0
            for s_next in range(num_states):
                future += transitions[s][a][s_next] * values[s_next]

            q += gamma * future

            if q > best:
                best = q

        new_values.append(best)

    return new_values