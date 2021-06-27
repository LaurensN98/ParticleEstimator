from pointProcesses import PointProcess
import numpy as np
import time

# Global variables
x_max = 1  # x-dimension of the figure
y_max = 1  # y-dimension of the figure
radii_mean = 0.08  # mean value of the radius of particles
radii_std = 0.005  # standard deviation of the radius of particles
offset_mean = 0  # mean value of the offset of the centers of the particles w.r.t. the boundary
offset_std = 0.02  # standard deviation of the offset of the centers of the particles w.r.t. the boundary
n_mean = 5000  # mean number of points of the poisson point process
rho = 0.3  # determines a relation between the intesities of each side of the poisson point process
precision = 8  # precision of the boundary finder
divisions = 15  # the number of divisions to make for approximating the particles
iterations = 5  # the number of iterations to approximate the particles


def full_estimation():
    # Create the Poisson process object
    poisson = PointProcess(
        width=x_max,
        height=y_max,
        bottom_seperation_point=np.random.uniform(0, x_max),
        top_seperation_point=np.random.uniform(0, x_max),
        add_particles=True,
        radii_mean=radii_mean,
        radii_std=radii_std,
        offset_mean=offset_mean,
        offset_std=offset_std)

    # Generates the actual Poisson process
    poisson.generate_points(
        n=n_mean,
        rho=rho,
        plot=False,
        show_info=False)

    poisson.find_bounds(precision=precision,
                        anim=False,
                        plot=True,
                        plot_true_bound=True,
                        plot_true_particles=True,
                        plot_particles_approximated=True,
                        show_info=False)

    poisson.estimate_particles(
        divisions=divisions,
        iterations=iterations,
        plot=True,
        show_anchor_points=True,
        show_info=False)

    # Approximates the boundary
    poisson.find_bounds(precision=precision,
                        anim=False,
                        plot=True,
                        plot_true_bound=True,
                        plot_true_particles=True,
                        plot_particles_approximated=True,
                        show_info=True)

    error = poisson.l2_error
    del poisson
    return error


def particle_estimation():
    poisson = PointProcess(
        width=x_max,
        height=y_max,
        bottom_seperation_point=np.random.uniform(0, x_max),
        top_seperation_point=np.random.uniform(0, x_max),
        add_particles=True,
        radii_mean=radii_mean,
        radii_std=radii_std,
        offset_mean=offset_mean,
        offset_std=offset_std)

    # Generates the actual Poisson process
    poisson.generate_points(
        n=n_mean,
        rho=rho,
        plot=False,
        show_info=False)

    # Estimates the particles
    poisson.estimate_particles(
        divisions=divisions,
        iterations=iterations,
        plot=True,
        show_anchor_points=True,
        show_info=True)


if __name__ == '__main__':
    particle_estimation()
