import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

class Genetic_optimization:
    def __init__(self, func, pop_num_max, top_best, x_min, x_max, y_min, y_max, chrom_lenght, prob_mutation, minimize = True, method = '1 point'):
        self.func = func
        self.pop_num_max = pop_num_max
        self.top_best = top_best
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.chrom_lenght = chrom_lenght
        self.prob_mutation = prob_mutation
        self.minimize = minimize
        self.method = method
    def create_first_population(self):
        f_pop = []
        for i in range(self.top_best**2):
            chromosome = ""
            for j in range(self.chrom_lenght):
                chromosome += str(random.randrange(0, 2))
            f_pop.append(chromosome)
        return f_pop
    def bin_to_dec_a_to_b(self, chromosome, x = True):
        if x == True:
            return int(chromosome, 2)*(self.x_max - self.x_min)/(2**self.chrom_lenght - 1) + self.x_min
        else:
            return int(chromosome, 2)*(self.y_max - self.y_min)/(2**self.chrom_lenght - 1) + self.y_min
    def fitness_x_and_y(self, chromosome_x, chromosome_y):
        fit = self.func(self.bin_to_dec_a_to_b(chromosome_x, True), self.bin_to_dec_a_to_b(chromosome_y, False))
        return fit
    def crossover(self, p1, p2):
        if self.method == '1 point':
            divide_by = random.randrange(1, self.chrom_lenght)
            child1, child2 = p1[:divide_by] + p2[divide_by:], p2[:divide_by] + p1[divide_by:]
        elif self.method == '2 points':
            divide_by_1 = random.randrange(2, self.chrom_lenght//2)
            divide_by_2 = random.randrange(1, divide_by_1)
            child1, child2 = p1[:divide_by_1] + p2[divide_by_1:divide_by_2] + p1[divide_by_2:], p2[:divide_by_1] + p1[divide_by_1:divide_by_2] + p2[divide_by_2:]
        elif self.method == 'uniform':
            num_to_change = random.randrange(1, self.chrom_lenght)
            indexes = random.sample(range(1, self.chrom_lenght), num_to_change)
            child1 = list(p1)
            child2 = list(p2)
            for i in range(self.chrom_lenght):
                if i in indexes:
                    child1[i], child2[i] = child2[i], child1[i]
            child1 = ''.join(child1)
            child2 = ''.join(child2)
        return child1, child2
    def mutation(self, chromosome):
        gene = random.randrange(0, self.chrom_lenght)
        chromosome = chromosome[:gene] + '0' + chromosome[gene+1:] if chromosome[gene] == '1' else chromosome[:gene] + '1' + chromosome[gene+1:]
        return chromosome
    def birth_children(self, population_x, population_y):
        new_population = []
        fitnesses = {}
        for i in range(min(len(population_x), len(population_y))):
            fitnesses[self.fitness_x_and_y(population_x[i], population_y[i])] = population_x[i]
        sorted_fitnesses = dict(sorted(fitnesses.items()))
        if self.minimize == True:
            parents = list(sorted_fitnesses.values())[0:self.top_best]
        else:
            parents = list(sorted_fitnesses.values())[-self.top_best:]
        for i in range(len(parents)):
            new_population.append(parents[i])
        pairs = [[parents[i], parents[j]] for i in range(len(parents)) for j in range(i + 1, len(parents))]
        for i in range(len(pairs)):
            child1, child2 = self.crossover(pairs[i][0], pairs[i][1])
            chance = random.uniform(0, 1)
            child = random.randrange(0, 1)
            if chance <= self.prob_mutation:
                if child == 0:
                    child1 = self.mutation(child1)
                else:
                    child2 = self.mutation(child2)
            new_population.append(child1)
            new_population.append(child2)
        return new_population

    def iter(self, population_x, population_y):
        population_x, population_y = self.birth_children(population_x, population_y), self.birth_children(population_y, population_x)
        dec_x, dec_y = [], []
        for i in range(len(population_x)):
            dec_x.append(self.bin_to_dec_a_to_b(population_x[i], True))
            dec_y.append(self.bin_to_dec_a_to_b(population_y[i], False))
        best_x, best_y = dec_x[0], dec_y[0]
        if self.minimize == True:
            for i in range(len(dec_x)):
                for j in range(len(dec_y)):
                    if self.func(dec_x[i], dec_y[j]) <= self.func(best_x, best_y):
                        best_x, best_y = dec_x[i], dec_y[j]
        else:
            for i in range(len(dec_x)):
                for j in range(len(dec_y)):
                    if self.func(dec_x[i], dec_y[j]) >= self.func(best_x, best_y):
                        best_x, best_y = dec_x[i], dec_y[j]
        return population_x, population_y, best_x, best_y
    def optimize_x_and_y(self):
        pop_num = 1
        population_x, population_y = self.create_first_population(), self.create_first_population()
        while pop_num != self.pop_num_max:
            population_x, population_y, best_x, best_y = self.iter(population_x, population_y)
            pop_num += 1
        return best_x, best_y, self.func(best_x, best_y)
    def plot_x_and_y(self):
        x, y, z = self.optimize_x_and_y()
        fig, ax = plt.subplots(subplot_kw = {"projection": "3d"})
        X, Y = np.arange(self.x_min, self.x_max, 0.25), np.arange(self.y_min, self.y_max, 0.25)
        X, Y = np.meshgrid(X, Y)
        Z = self.func(X, Y)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.25)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_zlim(Z.min(),Z.max())
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter('{x:.02f}')
        ax.scatter(x, y, z, c = 'orchid')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        return x, y, z
    def optimize_and_store_x_and_y(self):
        pop_num = 1
        pop_x, pop_y, pop_z = [], [], []
        bests_x, bests_y, bests_z = [], [], []
        population_x, population_y = self.create_first_population(), self.create_first_population()
        tmp_x, tmp_y, tmp_z = [], [], []
        for i in range(len(population_x)):
            tmp_x.append(self.bin_to_dec_a_to_b(population_x[i], True))
            tmp_y.append(self.bin_to_dec_a_to_b(population_y[i], False))
            tmp_z.append(self.func(tmp_x[-1], tmp_y[-1]))
        pop_x.append(tmp_x)
        pop_y.append(tmp_y)
        pop_z.append(tmp_z)
        bests_x.append(tmp_x[0])
        bests_y.append(tmp_y[0])
        bests_z.append(self.func(tmp_x[0], tmp_y[0]))
        while pop_num != self.pop_num_max:
            tmp_x, tmp_y, tmp_z = [], [], []
            population_x, population_y, best_x, best_y = self.iter(population_x, population_y)
            for i in range(len(population_x)):
                tmp_x.append(self.bin_to_dec_a_to_b(population_x[i], True))
                tmp_y.append(self.bin_to_dec_a_to_b(population_y[i], False))
                tmp_z.append(self.func(tmp_x[-1], tmp_y[-1]))
            pop_x.append(tmp_x)
            pop_y.append(tmp_y)
            pop_z.append(tmp_z)
            bests_x.append(best_x)
            bests_y.append(best_y)
            bests_z.append(self.func(best_x, best_y))
            pop_num += 1
        return pop_x, pop_y, pop_z, bests_x, bests_y, bests_z
    def plot_dynamic(self):
        pop_x, pop_y, pop_z, bests_x, bests_y, bests_z = self.optimize_and_store_x_and_y()
        X, Y = np.arange(self.x_min, self.x_max, 0.25), np.arange(self.y_min, self.y_max, 0.25)
        X, Y = np.meshgrid(X, Y)
        Z = self.func(X, Y)
        def update_graph(num):
            all_pop._offsets3d = (pop_x[num], pop_y[num], pop_z[num])
            best._offsets3d = ([bests_x[num]], [bests_y[num]], [bests_z[num]])
            title.set_text('Genetic optimization, iteration {}'.format(num+1))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        title = ax.set_title('Genetic optimization')
        all_pop = ax.scatter([], [], [], c = 'cornflowerblue', zorder = 1)
        best = ax.scatter([], [], [], c = 'orchid', zorder = 1) 
        surf = ax.plot_surface(X, Y, Z, cmap = cm.viridis, linewidth=0, antialiased = False, alpha=0.25)
        ani = matplotlib.animation.FuncAnimation(fig, update_graph, self.pop_num_max, interval=60, blit=False, repeat=False)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        #name = "/Users/anastasiiadenysenko/Desktop/genetic_" + list(str(self.func).split(' '))[1] + ".gif"
        #ani.save(filename=name, dpi=300, writer=PillowWriter(fps=25))
    def plot_bests(self):
        _, _, _, _, _, bests_z = self.optimize_and_store_x_and_y()
        fig, ax = plt.subplots()
        ax.plot(list(i for i in range(len(bests_z))), bests_z,  c = 'cornflowerblue')
        plt.title("Bets values")
        plt.show()

def erkli(X, Y):
    return -20*np.e**(-0.2*np.sqrt(0.5*(X**2 + Y**2))) - np.e**(0.5*(np.cos(2*np.pi*X) + np.cos(2*np.pi*Y))) + np.e + 20
def branin(x1, x2):
    b = 5.1/(4*np.pi**2)
    c = 5/np.pi
    t = 1/(8*np.pi)
    return (x2 - b*x1**2 + c*x1 - 6)**2 + 10*(1-t)*np.cos(x1) + 10
def easom(x1, x2):
    return -np.cos(x1)*np.cos(x2)*np.e**(-(x1-np.pi)**2 - (x2-np.pi)**2)
def goldstein_price(x1, x2):
    return (1+(x1+x2+1)**2 * (19-14*x1+3*x1**2 - 14*x2 + 6*x1*x2 +3*x2**2))*(30+(2*x1 - 3*x2**2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))
def six_hump_camel(x1, x2):
    return (4-2.1*x1**2 + (x1**4)/3)*x1**2 + x1*x2 + (-4+4*x2**2)*x2**2

g = Genetic_optimization(erkli, 40, 7, -5, 5, -5, 5, 8, 0.01, minimize = True)
#g.plot_dynamic()
#g.plot_bests()
#g.plot_x_and_y()
g1 = Genetic_optimization(branin, 40, 10, 0, 10, 0, 15, 8, 0.1, minimize = True, method = 'uniform')
#g1.plot_dynamic()
#g1.plot_bests()
#g1.plot_x_and_y()
g2 = Genetic_optimization(easom, 40, 7, -10, 10, -10, 10, 8, 0.01, minimize = True)
#g2.plot_dynamic()
#g2.plot_bests()
#g2.plot_x_and_y()
g3 = Genetic_optimization(goldstein_price, 50, 7, -2, 2, -2, 2, 8, 0.1, minimize = False, method = '2 points')
#g3.plot_dynamic()
#g3.plot_bests()
#g3.plot_x_and_y()
g4 = Genetic_optimization(six_hump_camel, 60, 5, -3, 3, -2, 2, 8, 0.1, minimize = False)
#g4.plot_dynamic()
#g4.plot_bests()
#g4.plot_x_and_y()


def do_analysis(func, pop_num_max, top_best, prob_mutation, x_min, x_max, y_min, y_max, chrom_lenght, method, minimize = True):
    fig, ax = plt.subplots()
    if type(pop_num_max) == list:
        for i in range(len(pop_num_max)):
            genetic = Genetic_optimization(func, pop_num_max[i], top_best, x_min, x_max, y_min, y_max, chrom_lenght, prob_mutation, minimize, method)
            _, _, _, _, _, bests_z = genetic.optimize_and_store_x_and_y()
            ax.plot(list(i for i in range(len(bests_z))), bests_z, label = pop_num_max[i])
        ax.legend()
        plt.title("Number of iterations (minimum)" if minimize==True else "Number of iterations (maximum)")
        plt.show()
        return
    elif type(top_best) == list:
        for i in range(len(top_best)):
            genetic = Genetic_optimization(func, pop_num_max, top_best[i], x_min, x_max, y_min, y_max, chrom_lenght, prob_mutation, minimize, method)
            _, _, _, _, _, bests_z = genetic.optimize_and_store_x_and_y()
            ax.plot(list(i for i in range(len(bests_z))), bests_z, label = top_best[i])
        ax.legend()
        plt.title("Number of chromosomes that qualify for being parents (minimum)" if minimize==True else "Number of chromosomes that qualify for being parents (maximum)")
        plt.show()
        return
    elif type(prob_mutation) == list:
        for i in range(len(prob_mutation)):
            genetic = Genetic_optimization(func, pop_num_max, top_best, x_min, x_max, y_min, y_max, chrom_lenght, prob_mutation[i], minimize, method)
            _, _, _, _, _, bests_z = genetic.optimize_and_store_x_and_y()
            ax.plot(list(i for i in range(len(bests_z))), bests_z, label = prob_mutation[i])
        ax.legend()
        plt.title("Probability of mutation (minimum)" if minimize==True else "Probability of mutation (maximum)")
        plt.show()
        return
    elif type(chrom_lenght) == list:
        for i in range(len(chrom_lenght)):
            genetic = Genetic_optimization(func, pop_num_max, top_best, x_min, x_max, y_min, y_max, chrom_lenght[i], prob_mutation, minimize, method)
            _, _, _, _, _, bests_z = genetic.optimize_and_store_x_and_y()
            ax.plot(list(i for i in range(len(bests_z))), bests_z, label = chrom_lenght[i])
        ax.legend()
        plt.title("Lenght of chromosome (minimum)" if minimize==True else "Lenght of chromosome (maximum)")
        plt.show()
        return
    elif type(method) == list:
        for i in range(len(method)):
            genetic = Genetic_optimization(func, pop_num_max, top_best, x_min, x_max, y_min, y_max, chrom_lenght, prob_mutation, minimize, method = method[i])
            _, _, _, _, _, bests_z = genetic.optimize_and_store_x_and_y()
            ax.plot(list(i for i in range(len(bests_z))), bests_z, label = method[i])
        ax.legend()
        plt.title("Type of crossover (minimum)" if minimize==True else "Type of crossover (maximum)")
        plt.show()
        return
#do_analysis(goldstein_price, 50, 7, 0.1, -2, 2, -2, 2, 8, method = ['1 point', '2 points', 'uniform'], minimize = False)