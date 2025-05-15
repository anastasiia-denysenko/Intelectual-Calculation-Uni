import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from matplotlib.ticker import LinearLocator
from copy import deepcopy
from matplotlib.animation import PillowWriter

class PSO_for_x_and_y(object):
    def __init__(self, func, num_iter, a1, a2, size_pop, v_min, v_max, x_min, x_max, y_min, y_max, minimize = True):
        if a1 < 0 or a1 > 4 or a2 < 0 or a2 > 4:
            raise ValueError("Accelerations has to be in range (0, 4), but given values are {i} and {j}".format(i = a1, j = a2))
        if v_max <= 0:
            raise ValueError("Maximum velocity has to be greater than 0, but given value is {i}".format(i = v_max))
        self.func = func
        self.num_iter = num_iter
        self.a1 = a1
        self.a2 = a2
        self.size_pop = size_pop
        self.v_min = v_min
        self.v_max = v_max
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.minimize = minimize
        self.curr_iter = 0
        self.pop_x, self.pop_y, self.pop_z = self.create_first_population()
        self.x_bests, self.y_bests, self.z_bests, self.x_ideal, self.y_ideal, self.z_ideal = self.find_fitness()
        self.vel_x, self.vel_y = self.create_velocity_vector()
        self.diff_x = abs(self.x_max) + abs(self.x_min)
        self.diff_y = abs(self.y_max) + abs(self.y_min)
        self.all_z_ideal = [self.z_ideal]
    def put_in_boundry(self, i, j, x = True):
        if x == True: 
            if self.pop_x[-1][i][j] <= self.x_min or self.pop_x[-1][i][j] >= self.x_max:
                self.vel_x[i][j] *= -1
                self.pop_x[-1][i][j] =  ((self.pop_x[-1][i][j] - self.x_min) % self.diff_x + self.diff_x) % self.diff_x + self.x_min
        else:
            if self.pop_y[-1][i][j] <= self.y_min or self.pop_y[-1][i][j] >= self.y_max:
                self.vel_x[i][j] *= -1
                self.pop_y[-1][i][j] = ((self.pop_y[-1][i][j] - self.y_min) % self.diff_y + self.diff_y) % self.diff_y + self.y_min

    def create_first_population(self):
        self.pop_x = []
        self.pop_y = []
        stp_x = abs((self.x_min - self.x_max))/self.size_pop
        stp_y = abs((self.y_min - self.y_max))/self.size_pop
        p_x, p_y = np.arange(self.x_min, self.x_max, stp_x), np.arange(self.y_min, self.y_max, stp_y)
        p_x, p_y = np.meshgrid(p_x, p_y)
        p_x, p_y = p_x.tolist(), p_y.tolist()
        self.pop_x.append(list(p_x))
        self.pop_y.append(list(p_y))
        self.pop_z = deepcopy(self.pop_x)
        for i in range(self.size_pop):
            for j in range(self.size_pop):
                self.pop_z[-1][i][j] = self.func(self.pop_x[-1][i][j], self.pop_y[-1][i][j])
        return self.pop_x, self.pop_y, self.pop_z
    def find_fitness(self):
        if self.curr_iter == 0:
            self.x_ideal = self.x_min
            self.y_ideal = self.y_min
            self.z_ideal = self.func(self.x_min, self.y_min)
            self.x_bests = np.full((self.size_pop, self.size_pop), self.x_min)
            self.y_bests = np.full((self.size_pop, self.size_pop), self.y_min)
            self.z_bests = np.full((self.size_pop, self.size_pop), self.y_min)
        if self.minimize == True:
            for i in range(self.size_pop):
                for j in range(self.size_pop):
                    if self.func(self.pop_x[-1][i][j], self.pop_y[-1][i][j]) < self.func(self.x_bests[i][j], self.y_bests[i][j]):
                        self.x_bests[i][j] = self.pop_x[-1][i][j]
                        self.y_bests[i][j] = self.pop_y[-1][i][j]
                        self.z_bests[i][j] = self.func(self.x_bests[i][j], self.y_bests[i][j])
                        if self.func(self.pop_x[-1][i][j], self.pop_y[-1][i][j]) < self.z_ideal and self.pop_x[-1][i][j] >= self.x_min and self.pop_x[-1][i][j] <= self.x_max and self.pop_y[-1][i][j] >= self.y_min and self.pop_y[-1][i][j] <= self.y_max:
                            self.x_ideal = self.pop_x[-1][i][j]
                            self.y_ideal = self.pop_y[-1][i][j]
                            self.z_ideal = self.func(self.x_ideal, self.y_ideal)
        else:
            for i in range(self.size_pop):
                for j in range(self.size_pop):
                    if self.func(self.pop_x[-1][i][j], self.pop_y[-1][i][j]) > self.func(self.x_bests[i][j], self.y_bests[i][j]):
                        self.x_bests[i][j] = self.pop_x[-1][i][j]
                        self.y_bests[i][j] = self.pop_y[-1][i][j]
                        self.z_bests[i][j] = self.func(self.x_bests[i][j], self.y_bests[i][j])
                        if self.func(self.pop_x[-1][i][j], self.pop_y[-1][i][j]) > self.z_ideal and self.pop_x[-1][i][j] >= self.x_min and self.pop_x[-1][i][j] <= self.x_max and self.pop_y[-1][i][j] >= self.y_min and self.pop_y[-1][i][j] <= self.y_max:
                            self.x_ideal = self.pop_x[-1][i][j]
                            self.y_ideal = self.pop_y[-1][i][j]
                            self.z_ideal = self.func(self.x_ideal, self.y_ideal)
        return self.x_bests, self.y_bests, self.z_bests, self.x_ideal, self.y_ideal, self.z_ideal
    def create_velocity_vector(self):
        if self.curr_iter == 0:
            self.vel_x, self.vel_y = [], []
            for i in range(self.size_pop):
                tmp_x = []
                tmp_y = []
                for j in range(self.size_pop):
                    tmp_x.append(self.v_min + (self.v_max - self.v_min)*random.uniform(0, 1))
                    tmp_y.append(self.v_min + (self.v_max - self.v_min)*random.uniform(0, 1))
                self.vel_x.append(tmp_x)
                self.vel_y.append(tmp_y)
        else:
            r1 = np.random.rand()
            r2 = np.random.rand()
            for i in range(self.size_pop):
                for j in range(self.size_pop):
                    self.vel_x[i][j] += self.a1*(self.x_bests[i][j]-self.pop_x[-1][i][j])*r1 + self.a2*(self.x_ideal - self.pop_x[-1][i][j])*r2
                    self.vel_y[i][j] += self.a1*(self.y_bests[i][j]-self.pop_y[-1][i][j])*r1 + self.a2*(self.y_ideal - self.pop_y[-1][i][j])*r2
        return self.vel_x, self.vel_y
    def update_positions(self):
        self.pop_x.append(self.pop_x[-1])
        self.pop_y.append(self.pop_y[-1])
        self.pop_z.append(self.pop_z[-1])
        for i in range(self.size_pop):
             for j in range(self.size_pop):
                self.pop_x[-1][i][j] += self.vel_x[i][j]
                self.pop_y[-1][i][j] += self.vel_y[i][j]
                self.put_in_boundry(i, j, True)
                self.put_in_boundry(i, j, False)
                self.pop_z[-1][i][j] = self.func(self.pop_x[-1][i][j], self.pop_y[-1][i][j])
        return self.pop_x[-1], self.pop_y[-1]
    def iter(self):
        self.find_fitness()
        self.create_velocity_vector()
        self.update_positions()
        self.curr_iter += 1
    def run(self):
        while self.curr_iter != self.num_iter:
            self.iter()
            self.all_z_ideal.append(self.z_ideal)
        return self.pop_x[-1], self.pop_y[-1], self.x_ideal, self.y_ideal, self.z_ideal
    def plot(self):
        self.run()
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        X, Y = np.arange(self.x_min, self.x_max, 0.25), np.arange(self.y_min, self.y_max, 0.25)
        X, Y = np.meshgrid(X, Y)
        Z = self.func(X, Y)
        surf = ax.plot_surface(X, Y, Z, cmap=matplotlib.cm.viridis, linewidth=0, antialiased=False, alpha=0.25)
        ax.set_zlim(Z.min(), Z.max())
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter('{x:.02f}')
        ax.scatter(self.x_ideal, self.y_ideal, self.z_ideal, c = 'orchid')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    def run_and_plot(self):
        X, Y = np.arange(self.x_min, self.x_max, 0.25), np.arange(self.y_min, self.y_max, 0.25)
        X, Y = np.meshgrid(X, Y)
        Z = self.func(X, Y)
        def update_graph(num):
            self.iter()
            for i in range(self.size_pop):
                all_pop._offsets3d = (self.pop_x[-1][i], self.pop_y[-1][i], self.pop_z[-1][i])
            best._offsets3d = ([self.x_ideal], [self.y_ideal], [self.z_ideal])
            title.set_text('PSO, iteration {}'.format(num+1))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        title = ax.set_title('PSO')
        all_pop = ax.scatter(self.pop_x[0][0], self.pop_y[0][0], self.pop_z[0][0], c = 'cornflowerblue', zorder = 1)
        best = ax.scatter([self.x_min], [self.y_min], [self.func(self.x_min, self.y_min)], c = 'orchid', zorder = 1)
        surf = ax.plot_surface(X, Y, Z, cmap = matplotlib.cm.viridis, linewidth=0, antialiased = False, alpha=0.25)
        ani = matplotlib.animation.FuncAnimation(fig, update_graph, self.num_iter, interval=60, blit=False, repeat=False)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.axes.set_xlim3d(left=self.x_min*2, right=self.x_max*2)
        ax.axes.set_ylim3d(bottom=self.y_min*2, top=self.y_max*2) 
        plt.show()
        name = "/Users/anastasiiadenysenko/Desktop/pso_" + list(str(self.func).split(' '))[1] + ".gif"
        ani.save(filename=name, dpi=300, writer=PillowWriter(fps=25))
    def plot_analysis(self):
        self.run()
        fig, ax = plt.subplots()
        ax.plot(list(i for i in range(len(self.all_z_ideal))), self.all_z_ideal)
        plt.title("Z-values to minimize {func}".format(func = list(str(self.func).split(' '))[1]) if self.minimize==True else "Z-values to maximize {func}".format(func = list(str(self.func).split(' '))[1]))
        plt.show()

def do_analysis(func, num_iter, a1, a2, size_pop, v_min, v_max, x_min, x_max, y_min, y_max, minimize = True):
    fig, ax = plt.subplots()
    if type(a1) == list:
        for i in range(len(a1)):
            pso = PSO_for_x_and_y(func, num_iter, a1[i], a2, size_pop, v_min, v_max, x_min, x_max, y_min, y_max, minimize = True)
            pso.run()
            ax.plot(list(i for i in range(len(pso.all_z_ideal))), pso.all_z_ideal, label = a1[i])
        ax.legend()
        plt.title("First aceleration (minimum)" if minimize==True else "First aceleration (maximum)")
        plt.show()
        return
    elif type(a2) == list:
        for i in range(len(a2)):
            pso = PSO_for_x_and_y(func, num_iter, a1, a2[i], size_pop, v_min, v_max, x_min, x_max, y_min, y_max, minimize = True)
            pso.run()
            ax.plot(list(i for i in range(len(pso.all_z_ideal))), pso.all_z_ideal, label = a2[i])
        ax.legend()
        plt.title("Second aceleration (minimum)" if minimize==True else "Second aceleration (maximum)")
        plt.show()
        return
    elif type(size_pop) == list:
        for i in range(len(size_pop)):
            pso = PSO_for_x_and_y(func, num_iter, a1, a2, size_pop[i], v_min, v_max, x_min, x_max, y_min, y_max, minimize = True)
            pso.run()
            ax.plot(list(i for i in range(len(pso.all_z_ideal))), pso.all_z_ideal, label = size_pop[i])
        ax.legend()
        plt.title("Population size (minimum)" if minimize==True else "Population size (maximum)")
        plt.show()
        return
    elif type(v_min) == list:
        for i in range(len(v_min)):
            pso = PSO_for_x_and_y(func, num_iter, a1, a2, size_pop, v_min[i], v_max, x_min, x_max, y_min, y_max, minimize = True)
            pso.run()
            ax.plot(list(i for i in range(len(pso.all_z_ideal))), pso.all_z_ideal, label = v_min[i])
        ax.legend()
        plt.title("Minimum velocity (minimum)" if minimize==True else "Minimum velocity (maximum)")
        plt.show()
        return
    elif type(v_max) == list:
        for i in range(len(v_max)):
            pso = PSO_for_x_and_y(func, num_iter, a1, a2, size_pop, v_min, v_max[i], x_min, x_max, y_min, y_max, minimize = True)
            pso.run()
            ax.plot(list(i for i in range(len(pso.all_z_ideal))), pso.all_z_ideal, label = v_max[i])
        ax.legend()
        plt.title("Maximum velocity (minimum)" if minimize==True else "Maximum velocity (maximum)")
        plt.show()
        return
def erkli(X, Y):
    return -20*np.e**(-0.2*np.sqrt(0.5*(X**2 + Y**2))) - np.e**(0.5*(np.cos(2*np.pi*X) + np.cos(2*np.pi*Y))) + np.e + 20
def rozenbrok(x, y):
    return (1-x)**2 + 100*(y-x**2)**2
def cross_tray(x, y):
    n = np.e ** (abs(100 - ((x**2 + y**2)**(1/2)/np.pi)))
    return (abs(np.sin(x)*np.sin(y)*n)+1)**0.1
def holder_table(x, y):
    return -abs(np.sin(x)*np.cos(y)*np.e**((x**2 + y**2)**(1/2)/np.pi))
def mccormic(x, y):
    return np.sin(x+y) + (x-y)**2 - 1.5*x + 2.5*y + 1
p = PSO_for_x_and_y(holder_table, 50, 2, 2, 20, -1, 3, -10, 10, -10, 10, minimize = True) 
#p.plot_analysis()   
#p.run_and_plot()
p1 = PSO_for_x_and_y(erkli, 50, 2, 2, 20, -1, 3, -5, 5, -5, 5, minimize = True)
#p1.plot_analysis()   
#p1.run_and_plot()
p2 = PSO_for_x_and_y(rozenbrok, 50, 2, 2, 20, -1, 3, -2, 2, -0.5, 3, minimize = True)
#p2.run_and_plot()
#p2.plot_analysis()   
p3 = PSO_for_x_and_y(cross_tray, 50, 2, 2, 20, -1, 3, -10, 10, -10, 10, minimize = True)    
#p3.run_and_plot()
#p3.plot_analysis()   
p4 = PSO_for_x_and_y(mccormic, 50, 2, 2, 20, -1, 3, -1.5, 4, -3, 4, minimize = True)    
#p4.run_and_plot()
#p4.plot_analysis()   
#do_analysis(mccormic, 50, 2, 2, 20, -1, [2, 4, 6, 7], -1.5, 4, -3, 4, minimize = True)