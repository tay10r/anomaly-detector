from src.task import Task

class Loop:
    def __init__(self):
        self.tasks: list[Task] = []

    def add_task(self, task: Task):
        self.tasks.append(task)

    def step(self):
        for task in self.tasks:
            task.step()