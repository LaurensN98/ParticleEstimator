# Needed imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import itertools
from IPython import display as dp
import time
from structs import Point, Particle, Section, Distance
from functions import find_area_segment, point_in_circle, distance_points, l2, in_particle


class PointProcess:
    def __init__(self, width=1, height=1, bottom_seperation_point=0.3, top_seperation_point=0.7, add_particles=True,
                 radii_mean=0.08, radii_std=0.005, offset_mean=0, offset_std=0.02):
        self.width = width
        self.height = height
        self.b = bottom_seperation_point
        self.t = top_seperation_point
        self.particles = []
        self.particles_approximated = []
        self.l2_error = None
        self.points = []
        self.parameter_left = 0
        self.parameter_right = 0
        self.n_left = 0
        self.n_right = 0
        self.b_approximated = 0
        self.t_approximated = 0
        self.squared_error = 0
        self.radii_mean = radii_mean
        self.radii_std = radii_std
        self.offset_mean = offset_mean
        self.offset_std = offset_std

        # Assigns the slope between the bottom_seperation_point and top_seperation_point to an attribute.
        if self.t != self.b:
            self.slope = height / (self.t - self.b)
        # If the bottom_seperation_point equals the top_seperation_point, we have that the slope is vertical.
        else:
            self.slope = 1000000000000

        # If add_particles is true, particles are created and points within these particles are deleted from
        # list self.points.
        if add_particles:
            y_values = [max(0, 0.2 + 0.2 * np.random.randn())]
            radii = [self.radii_mean + self.radii_std * np.random.randn()]
            test = y_values[0]
            i = 0
            while test < self.height:
                test += radii[i] + max(0, 0.1 + 0.01 * np.random.randn())
                radii += [self.radii_mean + self.radii_std * np.random.randn()]
                test += radii[i + 1]
                y_values += [test]
                i += 1

            # Prevents particles from lying outside the boundaries.
            if y_values[0] - radii[0] < 0:
                y_values = y_values[1:]
                radii = radii[1:]
            while y_values[-1] + radii[-1] > self.height:
                y_values = y_values[:-1]
                radii = radii[:-1]

            # Compute the x_values for the corresponding y_values and pass the x, y, and radius information on to
            # an Particle object.
            x_values = [(y / self.slope + self.b + self.offset_mean + self.offset_std * np.random.randn()) for y in
                        y_values]
            self.particles += [Particle(x_values[j], y_values[j], radii[j]) for j in range(len(x_values))]

        # Calculates the surface of the individual segments based on bottom_seperation_point and top_seperation_point.
        # The areas of the particles is deducted from the individual segments since the density is zero in these regions
        self.surface_left = (min(self.b, self.t) + (np.abs(self.b - self.t) / 2)) * self.height
        self.surface_right = ((self.width - max(self.b, self.t)) + (np.abs(self.b - self.t) / 2)) * self.height
        if self.particles:
            for i, particle in enumerate(self.particles):
                self.surface_left -= (
                        find_area_segment(self.b, self.t, particle, self.height, "left") - find_area_segment(0, 0,
                                                                                                             particle,
                                                                                                             self.height,
                                                                                                             "left"))
                self.surface_right -= (
                        find_area_segment(self.b, self.t, particle, self.height, "right") - find_area_segment(
                    self.width, self.width, particle, self.height, "right"))

    def generate_points(self, n=10000, rho=0.3, plot=True, show_info=True):

        # Logs start time of script execution to keep track of execution time.
        start_time = time.time()

        # Empty self.points list in case script is run twice.
        self.points = []

        self.parameter_left = n / (self.surface_left + rho * self.surface_right)
        self.parameter_right = rho * self.parameter_left

        self.n_left = self.parameter_left * self.surface_left
        self.n_right = self.parameter_right * self.surface_right

        # Simulates the number of points in the left segment following a poisson distribution with parameter_left.
        # For each point a Point object is made and while the point is not in the left segment,
        # it will be resampled until it is.
        num_of_points = np.random.poisson(self.n_left)
        for point in range(num_of_points):
            test = Point(np.random.uniform(0, self.width), np.random.uniform(0, self.height))
            while test.x > test.y / self.slope + self.b or in_particle(self.particles, test):
                test = Point(np.random.uniform(0, self.width), np.random.uniform(0, self.height))
            self.points += [test]

        # Similar as above
        num_of_points = np.random.poisson(self.n_right)
        for point in range(num_of_points):
            test = Point(np.random.uniform(0, self.width), np.random.uniform(0, self.height))
            while test.x < test.y / self.slope + self.b or in_particle(self.particles, test):
                test = Point(np.random.uniform(0, self.width), np.random.uniform(0, self.height))
            self.points += [test]

        if plot:
            self.plot()

        execution_time = (time.time() - start_time)
        if show_info:
            print("Number of points:", len(self.points))
            print("Execution time: {0} seconds".format(execution_time))

    # __________________________________________TEST_________________________________________________

    def likelihood(self, b, t):
        surface_left = (min(b, t) + (np.abs(b - t) / 2)) * self.height
        surface_right = ((self.width - max(b, t)) + (np.abs(b - t) / 2)) * self.height
        if self.particles_approximated:
            for i, particle in enumerate(self.particles_approximated):
                surface_left -= (
                        find_area_segment(b, t, particle, self.height, "left") - find_area_segment(0, 0, particle,
                                                                                                   self.height,
                                                                                                   "left"))
                surface_right -= (
                        find_area_segment(b, t, particle, self.height, "right") - find_area_segment(self.width,
                                                                                                    self.width,
                                                                                                    particle,
                                                                                                    self.height,
                                                                                                    "right"))

        if t != b:
            slope = self.height / (t - b)
        else:
            slope = 1000000000000
        num_of_points_a = sum([int(i.x < i.y / slope + b) for i in self.points])
        num_of_points_b = len(self.points) - num_of_points_a
        loglikelihood_a = -(surface_left * self.parameter_left) + num_of_points_a * np.log(
            surface_left * self.parameter_left) - sum([np.log(i + 1) for i in range(num_of_points_a)])
        loglikelihood_b = -(surface_right * self.parameter_right) + num_of_points_b * np.log(
            surface_right * self.parameter_right) - sum([np.log(j + 1) for j in range(num_of_points_b)])
        loglikelihood = loglikelihood_a + loglikelihood_b
        return loglikelihood

    def find_bounds_single(self, bottom_interval, top_interval, anim):
        spacing = (bottom_interval[1] - bottom_interval[0]) / 10
        lines = list(itertools.product(np.linspace(bottom_interval[0], bottom_interval[1], 11)[1:10],
                                       np.linspace(top_interval[0], top_interval[1], 11)[1:10]))
        best_likelihood = self.likelihood(lines[0][0], lines[0][1])
        best_line = 0
        for i, line in enumerate(lines[1:]):
            likeli = self.likelihood(line[0], line[1])
            if likeli > best_likelihood:
                best_likelihood = likeli
                best_line = i + 1
                # If anim is true, plot the bound and clear right after to simulate moving image. Turn anim of for
                # best performance.
                if anim:
                    self.b_approximated, self.t_approximated = line[0], line[1]
                    self.plot(plot_true_bound=False, plot_bound_approximated=True, plot_true_particles=False,
                              plot_particles_approximated=False, print_error=False)
                    dp.clear_output(wait=True)
        return [lines[best_line][0] - spacing, lines[best_line][0] + spacing], [lines[best_line][1] - spacing,
                                                                                lines[best_line][1] + spacing]

    def find_bounds(self, precision=3, anim=False, plot=False, plot_true_bound=False, plot_true_particles=False,
                    plot_particles_approximated=False, show_info=True):
        start_time = time.time()
        error_memory = None

        if self.l2_error:
            error_memory = self.l2_error
        # Set the original interval of values on which to check the likelihood function.
        bottom_interval, top_interval = [0, self.width], [0, self.width]
        # precision defines how many times the function find_bounds is to be repeated. After each iteration a smaller
        # interval is used.
        for i in range(precision):
            bottom_interval, top_interval = self.find_bounds_single(bottom_interval, top_interval, anim)
        # After all iterations are done, set the approximated bound.
        self.b_approximated, self.t_approximated = (bottom_interval[1] + bottom_interval[0]) / 2, (
                top_interval[1] + top_interval[0]) / 2

        # Calculate the error values of this bound. squared_error sums the differences of the seperation points
        # squared while l2_error calculates the area between the two vectors.
        self.squared_error = (self.b_approximated - self.b) ** 2 + (self.t_approximated - self.t) ** 2
        self.l2_error = l2(self.b_approximated, self.t_approximated, self.b, self.t, self.height)
        if plot:
            self.plot(plot_true_bound=plot_true_bound, plot_bound_approximated=True,
                      plot_true_particles=plot_true_particles, plot_particles_approximated=plot_particles_approximated,
                      print_error=show_info)

        execution_time = (time.time() - start_time)

        if show_info:
            if error_memory:
                print("The error has been improved by {0} times".format(error_memory / self.l2_error))
            print("Execution time: {0} seconds".format(execution_time))

    def estimate_particles(self, divisions=10, iterations=3, plot=True,
                           show_anchor_points=False, show_info=True):
        start_time = time.time()
        # By appointing an empty list to self.particles we clear it to prevent double particles
        self.particles_approximated = []

        smallest_parameter = min(self.parameter_left, self.parameter_right)
        expected = (self.width / divisions) * (self.height / divisions) * smallest_parameter
        t = np.floor(0.83 * expected)

        xsect = np.arange(0, self.width, self.width / divisions)
        ysect = np.arange(0, self.height, self.height / divisions)
        blocks = list(itertools.product(xsect, ysect))

        sections = []
        for block in blocks:
            sections += [
                Section(block[0], block[0] + self.width / divisions, block[1], block[1] + self.height / divisions)]

        for point in self.points:
            for section in sections:
                if section.x_min < point.x < section.x_max and section.y_min < point.y < section.y_max:
                    section.add_point(point)
                    break

        particle_origin_points = []
        for section in sections:
            if len(section.points) < t:
                particle_origin_points += [
                    Point((section.x_min + section.x_max) / 2, (section.y_min + section.y_max) / 2)]

        if show_anchor_points:
            figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
            plt.plot([self.points[i].x for i in range(len(self.points))],
                     [self.points[i].y for i in range(len(self.points))], 'o', color='black', markersize=2)
            plt.plot([particle_origin_points[i].x for i in range(len(particle_origin_points))],
                     [particle_origin_points[i].y for i in range(len(particle_origin_points))], 'o', color='blue',
                     markersize=4)
            # for i in range(1, 10):
            #     plt.plot([0, 1], [i / 10, i / 10], color='Blue', markersize=4)
            # for i in range(1, 10):
            #     plt.plot([i / 10, i / 10], [0, 1], color='Blue', markersize=4)
            plt.ylim(0, self.height)
            plt.xlim(0, self.width)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()

        for anchor_point in particle_origin_points:
            double = False

            check = []
            for point in self.points:
                if point_in_circle(anchor_point, 0.3 * self.width, [point]):
                    check += [point]

            particle = self.calculate_center_and_radius_recursive(anchor_point, iterations, check)
            if self.particles_approximated:
                for particle_i in self.particles_approximated:
                    if point_in_circle(particle_i, particle_i.r, [particle]):
                        double = True
                        break
            if double:
                continue
            if (-3.11 * self.radii_std + self.radii_mean) < particle.r < (
                    3.11 * self.radii_std + self.radii_mean) and 0 < particle.x < self.width and 0 < particle.y < self.height:
                self.particles_approximated += [particle]

        figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
        if plot:
            self.plot(plot_true_particles=True, plot_particles_approximated=True)

        execution_time = (time.time() - start_time)

        if show_info:
            print("{0} particles have been detected.".format(len(self.particles_approximated)))
            print("Execution time: {0} seconds".format(execution_time))

    def calculate_center_and_radius_recursive(self, anchor_point, iterations, check):
        #         print(anchor_point.x, anchor_point.y)
        for i in range(iterations):
            prev_anchor_point = anchor_point
            if anchor_point.x < 0 or anchor_point.x > self.width or anchor_point.y < 0 or anchor_point.y > self.height:
                break
            anchor_point, r = self.calculate_center_and_radius(anchor_point, check)
            if distance_points(prev_anchor_point, anchor_point) > 0.3 * self.width:
                return Particle(-10, -10, 1)
        return Particle(anchor_point.x, anchor_point.y, r)

    def calculate_center_and_radius(self, anchor_point, check):
        first, second, third, reserve = self.closest_points(anchor_point, check)

        if np.abs((first.y - second.y) * (third.x - first.x) + (first.y - third.y) * (first.x - second.x)) == 0:
            third = reserve
        y = (1 / 2) * (((second.x - first.x) * (third.x ** 2 - first.x ** 2 + third.y ** 2 - first.y ** 2) + (
                third.x - first.x) * (first.x ** 2 - second.x ** 2 + first.y ** 2 - second.y ** 2)) / (
                               (first.y - second.y) * (third.x - first.x) + (first.y - third.y) * (
                               first.x - second.x)))

        if np.abs(second.x - first.x) == 0:
            x = (1 / 2) * ((third.x ** 2 - first.x ** 2 + 2 * y * (first.y - third.y) + third.y ** 2 - first.y ** 2) / (
                    third.x - first.x))
        else:
            x = (1 / 2) * (
                    (second.x ** 2 - first.x ** 2 + 2 * y * (first.y - second.y) + second.y ** 2 - first.y ** 2) / (
                    second.x - first.x))

        r = distance_points(Point(x, y), Point(first.x, first.y))
        center = Point(x, y)
        return center, r

    def closest_points(self, anchor_point, check):
        first = second = third = fourth = Distance(0, 0, max(self.width, self.height) * 100)
        for point in check:
            d = distance_points(anchor_point, point)
            if point.x < anchor_point.x and point.y > anchor_point.y:
                if d < first.d:
                    first = Distance(point.x, point.y, d)
                continue
            if point.x < anchor_point.x and point.y < anchor_point.y:
                if d < second.d:
                    second = Distance(point.x, point.y, d)
                continue
            if point.x > anchor_point.x and point.y > anchor_point.y:
                if d < third.d:
                    third = Distance(point.x, point.y, d)
                continue
            if point.x > anchor_point.x and point.y < anchor_point.y:
                if d < fourth.d:
                    fourth = Distance(point.x, point.y, d)
                continue

        circle_points = [first, second, third, fourth]
        combinations = list(itertools.combinations([1, 2, 3, 4], 3))
        argmax = np.argmax([sum([distance_points(circle_points[i - 1], circle_points[j - 1]) for i, j in
                                 list(itertools.combinations(combination, 2))]) for combination in combinations])

        for i in range(4):
            if i not in [combinations[argmax][0] - 1, combinations[argmax][1] - 1, combinations[argmax][2] - 1]:
                reserve = circle_points[i]
                break
        return circle_points[combinations[argmax][0] - 1], circle_points[combinations[argmax][1] - 1], circle_points[
            combinations[argmax][2] - 1], reserve

    def plot(self, plot_true_bound=False, plot_bound_approximated=False, plot_true_particles=False,
             plot_particles_approximated=False, print_error=False):
        figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
        ax = plt.gca()
        plt.ylim(0, self.height)
        plt.xlim(0, self.width)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.plot([self.points[i].x for i in range(len(self.points))],
                 [self.points[i].y for i in range(len(self.points))], 'o', color='black', markersize=2)

        if plot_bound_approximated:
            plt.plot([self.b_approximated, self.t_approximated], [0, self.height], linewidth=4)

        if plot_true_bound:
            plt.plot([self.b, self.t], [0, self.height], linestyle='--', linewidth=4, color='red')

        if plot_true_particles:
            plt.plot([self.particles[i].x for i in range(len(self.particles))],
                     [self.particles[i].y for i in range(len(self.particles))], 'o', color='red', markersize=5,
                     fillstyle='none')
            for particle in self.particles:
                ax.add_patch(
                    plt.Circle((particle.x, particle.y), particle.r, color='red', fill=False, linestyle='--'))

        if plot_particles_approximated:
            for particle in self.particles_approximated:
                ax.add_patch(plt.Circle((particle.x, particle.y), particle.r, color='blue', fill=False))

        plt.show()

        if print_error:
            print("The error squared is given by:", "{:.2e}".format(self.squared_error))
            print("The L2 error is given by:", "{:.2e}".format(self.l2_error))
