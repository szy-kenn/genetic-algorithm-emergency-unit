import random
from math import sqrt
import time
import matplotlib.pyplot as plt
from tabulate import tabulate

class GeneticAlgorithm:
  def __init__(self, grid, max_gen, acceptance_threshold, random_iteration_limit=100):

    self.grid = grid
    self.max_gen = max_gen
    self.acceptance_threshold = acceptance_threshold
    self.random_iteration_limit = random_iteration_limit

    self.current_gen = 0

    self.p1 = None
    self.p2 = None
    self.c1 = None
    self.c2 = None
    self.p1_score = 0 
    self.p2_score = 0 

    self.created_solutions = []
    self.best_solution = None
    self.prev_best = None

    self.proposed_coordinates = []
    self.response_times = []

    self.fig = plt.figure(figsize=(12, 7)) 
    self.ax = self.fig.add_subplot(111)
    self.table_ax = self.fig.add_subplot(222)
    self.avg_costs = []
    self.best_costs = []

    self.table_ax.axis('off')


    self.table = self.table_ax.table(
      cellText= list(reversed(self.grid)), 
      # cellText= self.grid, 
      loc='center',
      cellLoc='center')

    self.table.scale(.5, 1.2)

  def _add_generated_solutions(self, *args: list):
    """
    Adds the solution to the list of already created solutions
    """

    for solution in args:
      stringified = "".join(str(gene) for gene in solution)
      if stringified not in self.created_solutions:
        self.created_solutions.append(stringified)

  def _get_distance(self, pt1, pt2):
    """
    Returns the distance of two points using the distance formula
    """

    return sqrt(((pt2[0] - pt1[0]) ** 2) + ((pt2[1] - pt1[1]) ** 2))

  def _get_response_time(self, fitness_score):
    return 1.7 + (3.4 * fitness_score)

  def start(self):

    while self.current_gen <= self.max_gen:

      if self.current_gen == 0:
        self.p1 = self.select()
        self.p2 = self.select()

      else:
        # select random Parent 2
        temp_p2 = self.select()
        better_parent = self.p1 if self.p1_score  < self.p2_score else self.p2

        count = 0
        while ((self.fitness(temp_p2) > (self.fitness(self.p1) * (1 + self.acceptance_threshold)))
                and count < self.random_iteration_limit):
          temp_p2 = self.select()
          count += 1

        if count != 100:
          # if the loop stops before reaching the maximum iteration, then the randomly created solution is a valid Parent 2
          self.p2 = temp_p2 if self.fitness(temp_p2) < self.fitness(better_parent) else better_parent
        else:
          if random.randint(0, 1) == 1:
            self.p2 = self.best_solution
          else:
            self.p2 = better_parent

      self.p1_score, self.p2_score = self.fitness(self.p1), self.fitness(self.p2)
      self.avg_costs.append((self.p1_score + self.p2_score) / 2)

      c1, c2 = self.crossover(self.p1, self.p2)
      self.c1, self.c2 = self.mutate(c1), self.mutate(c2)
      c1_score, c2_score = self.fitness(self.c1), self.fitness(self.c2)

      count = 0
      tmp_children = []
      while (((c1_score > (self.p1_score * (1 + self.acceptance_threshold)) and c1_score > (self.p2_score * (1 + self.acceptance_threshold))) or
             (c2_score > (self.p1_score * (1 + self.acceptance_threshold)) and c2_score > (self.p2_score * (1 + self.acceptance_threshold)))) and 
              count < self.random_iteration_limit):
        c1, c2 = self.crossover(self.p1, self.p2)
        self.c1, self.c2 = self.mutate(c1), self.mutate(c2)
        c1_score, c2_score = self.fitness(self.c1), self.fitness(self.c2)
        tmp_children.append(self.c1)
        tmp_children.append(self.c2)
        count += 1
        
      if count == self.random_iteration_limit:
        tmp_children.sort(key=self.fitness)
        self.c1 = tmp_children[0]
        self.c2 = tmp_children[1]

      # check if a new best solution was found
      if self.p1_score < self.fitness(self.best_solution) or self.p2_score < self.fitness(self.best_solution):
        self.best_solution = self.p1 if self.p1_score < self.p2_score else self.p2

      # add all generated solutions in created_solutions list
      self._add_generated_solutions(self.p1, self.p2, self.c1, self.c2)

      # get the better offspring (lower cost) and make it the Parent 1 for the next generation
      self.p1 = self.c1 if c1_score < c2_score else self.c2
      self.best_costs.append(self.fitness(self.best_solution))

      # increment generation
      self.current_gen += 1
      self.update_plots()

      self.response_times.append(self._get_response_time(self.fitness(self.best_solution)))
      self.proposed_coordinates.append(self.best_solution)

    plt.show()
    self.save()
    return self.best_solution
  
  def save(self):
    with open (f"log-{int(time.time())}.txt", "w") as f:
      f.write(
      tabulate(([idx, (coord[0], coord[1]), f"{round(cost, 3)}km", f"{round(rtime, 3)}m"] for (idx, coord), cost, rtime in zip(enumerate(self.proposed_coordinates), self.best_costs, self.response_times)), 
                    headers=["Generation", "Proposed Coordinates", "Cost Value", "Response Time"], 
                    stralign='center')
      )

  def select(self):
    _tmp = [random.randint(1, 10), random.randint(1, 10)]
    while "".join(str(gene) for gene in _tmp) in self.created_solutions:
      _tmp = [random.randint(1, 10), random.randint(1, 10)]

    return _tmp 

  def fitness(self, solution):

    if solution is None:
       return float("+inf")

    total_cost = 0

    for y, row in enumerate(self.grid, start=1):
      for x, freq in enumerate(row, start=1):
        total_cost += self._get_distance((solution), (x, y)) * freq

    return total_cost

  def crossover(self, parent1, parent2):
    child1 = [parent1[0], parent2[1]] 
    child2 = [parent2[0], parent1[1]]

    return child1, child2

  def mutate(self, solution):
    random_pos = random.randint(0, 1) 
    match solution[random_pos]:
      case 1:
        solution[random_pos] += 1
      case 10:
        solution[random_pos] -= 1
      case _:
        if random.randint(0, 1) == 0:
          solution[random_pos] -= 1
        else:
          solution[random_pos] += 1
    return solution

  def update_plots(self):
    self.ax.set_xlabel('Generation')
    self.ax.set_ylabel('Cost Value')
    self.ax.set_title(f'Cost Value vs. Generation\nDistance = {self.fitness(self.best_solution)} at {self.best_solution}')
    self.ax.plot(self.best_costs, color='#d72b2d')
    self.ax.plot(self.avg_costs, alpha=0.25, color='blue')

    self.table_ax.axis('off')

    if self.prev_best is not None and self.prev_best != self.best_solution:
      self.table[(10 - self.prev_best[1], self.prev_best[0] - 1)].set_facecolor("w")
    self.table[(10 - self.best_solution[1], self.best_solution[0] - 1)].set_facecolor("#d72b2d")
    self.prev_best = self.best_solution

    plt.pause(0.05)
    self.ax.cla()

grid = [
  [0, 6, 2, 8, 7, 1, 2, 1, 5, 3],
  [4, 5, 9, 6, 3, 9, 7, 6, 5, 10],
  [1, 5, 2, 1, 2, 8, 3, 3, 6, 2],
  [1, 4, 0, 6, 8, 4, 0, 1, 2, 1],
  [7, 5, 8, 2, 5, 2, 3, 9, 8, 2],
  [4, 7, 4, 9, 9, 8, 6, 5, 4, 2],
  [1, 7, 1, 6, 9, 3, 1, 9, 6, 9],
  [4, 1, 2, 1, 3, 8, 7, 8, 9, 1],
  [5, 5, 3, 4, 4, 6, 4, 1, 9, 1],
  [5, 2, 4, 8, 9, 0, 3, 3, 8, 7],
]

# grid = [
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [1, 7, 1, 6, 2, 0, 0, 0, 0, 0],
#   [4, 1, 2, 1, 3, 0, 0, 0, 0, 0],
#   [5, 5, 3, 0, 4, 0, 0, 0, 0, 0],
#   [5, 2, 4, 8, 0, 0, 0, 0, 0, 0],
# ]

max_gen = 100
acceptance_threshold = 0.1
GA = GeneticAlgorithm(grid, max_gen, acceptance_threshold)
best_solution = GA.start()