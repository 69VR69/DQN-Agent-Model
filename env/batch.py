class Batch:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.experiences = []

    def add(self, experience):
        if len(self.experiences) >= self.batch_size:
            self.experiences.pop(0)
        self.experiences.append(experience)

    def is_full(self):
        return len(self.experiences) >= self.batch_size

    # to string
    def __str__(self):
        res = ""
        for exp in self.experiences:
            res += (str(exp) +";\n\n")
        return res


class Experience:
    # static variable
    ID = 0

    # state is an array of float between -1 and 1, action is an int between 0 and 8, reward is a float, next_state is an array of float between -1 and 1, done is a boolean
    def __init__(self, state, action, reward, next_state, done):
        Experience.ID += 1
        self.number = Experience.ID
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    # to string
    def __str__(self):
        return (
            "Experience n°"
            + str(self.number)
            + ":\n\tstate: "
            + str(self.state)
            + ",\n\taction: "
            + str(self.action)
            + ",\n\treward: "
            + str(self.reward)
            + ",\n\tnext_state: "
            + str(self.next_state)
            + ",\n\tdone: "
            + str(self.done)
        )


# Test
# exp1 = Experience(1, 2, 3, 4, 5)
# exp2 = Experience(6, 7, 8, 9, 10)
# exp3 = Experience(11, 12, 13, 14, 15)
# acc = Batch(2)
# print(acc)
# acc.add(exp1)
# acc.add(exp2)
# print(acc)
# acc.add(exp3)
# print(acc)