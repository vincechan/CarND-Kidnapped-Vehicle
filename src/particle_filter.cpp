/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 20;

	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_t(theta, std[2]);

	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_t(gen);
		p.weight = 1.0;
		particles.push_back(p);

		weights.push_back(1.0);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_t(0, std_pos[2]);

	for (auto& p : particles) {
		if (yaw_rate == 0) {
			p.x = p.x + velocity * delta_t * cos(p.theta);
			p.y = p.y + velocity * delta_t * sin(p.theta);
		}
		else {
			p.x = p.x + velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
			p.y = p.y + velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
			p.theta = p.theta + yaw_rate * delta_t;
		}

		p.x += dist_x(gen);
		p.y += dist_y(gen);
		p.theta += dist_t(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	weights.clear();

	for (auto& p : particles) {

		p.associations.clear();
		p.sense_x.clear();
		p.sense_y.clear();

		std::vector<LandmarkObs> obs_landmarks;
		for (const auto& obs : observations) {
			// translate to global coordinates
			LandmarkObs obs_landmark;
			obs_landmark.x = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
			obs_landmark.y = obs.x * sin(p.theta) + obs.y * cos(p.theta) + p.y;

			// find closest global landmark
			double min = 99999999;
			for (const auto& landmark : map_landmarks.landmark_list) {
				double distance = dist(landmark.x_f, landmark.y_f, obs_landmark.x, obs_landmark.y);
				if (distance <= min) {
					obs_landmark.id = landmark.id_i;
					min = distance;
				}
			}

			obs_landmarks.push_back(obs_landmark);
			p.associations.push_back(obs_landmark.id);
			p.sense_x.push_back(map_landmarks.landmark_list[obs_landmark.id - 1].x_f);
			p.sense_y.push_back(map_landmarks.landmark_list[obs_landmark.id - 1].y_f);
		}

		// calculate weight
		double s_x = std_landmark[0];
		double s_y = std_landmark[1];
		p.weight = 1.0;
		for (auto& obs : obs_landmarks) {
			double x = map_landmarks.landmark_list[obs.id - 1].x_f;
			double y = map_landmarks.landmark_list[obs.id - 1].y_f;
			double mu_x = obs.x;
			double mu_y = obs.y;
			double t1 = (x - mu_x) * (x - mu_x) / (2 * s_x * s_x);
			double t2 = (y - mu_y) * (y - mu_y) / (2 * s_y * s_y);
			p.weight = p.weight * exp(-(t1 + t2)) / (2 * 3.1415 * s_x * s_y);
		}

		weights.push_back(p.weight);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	discrete_distribution<> ddist(weights.begin(), weights.end());
	
	std::vector<Particle> new_particles;
	for (int i = 0; i < num_particles; i++) {
		new_particles.push_back(particles[ddist(gen)]);
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
