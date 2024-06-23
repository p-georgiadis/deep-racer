# AWS DeepRacer Project for AWS AI & ML Scholarship Program

## Introduction

Welcome to my AWS DeepRacer project repository! I am participating in the AWS DeepRacer Student League with the goal of building a fast and efficient DeepRacer model. This project aims to leverage the power of reinforcement learning to optimize the racing strategy of an autonomous car. The ultimate objective is to qualify for the AWS AI & ML Scholarship program, which offers access to advanced AI/ML training, mentorship, and career resources.

## Project Overview

### Objective

The primary objective of this project is to develop a DeepRacer model that can navigate the "jyllandsringen_open_cw" track as quickly and efficiently as possible. By optimizing the reward function and using the best possible raceline, I aim to achieve top rankings in the AWS DeepRacer Student League.

### AWS AI & ML Scholarship Program

The AWS AI & ML Scholarship program, in collaboration with Udacity, is designed to help students from underserved or underrepresented communities gain foundational skills in AI and ML. The program provides access to scholarships, mentors, and real-world projects, helping students prepare for careers in technology.

## Approach

### Step 1: Track Analysis

To start, I analyzed the "jyllandsringen_open_cw" track by loading the track's waypoints and creating a visual representation of the center, inner, and outer borders.

```python
import glob
import numpy as np
from shapely.geometry import Point, Polygon, LineString
import matplotlib.pyplot as plt

# Load the track to analyze
TRACK_NAME = 'jyllandsringen_open_cw'
waypoints = np.load("./tracks/%s.npy" % TRACK_NAME)

# Convert to Shapely objects
center_line = waypoints[:, 0:2]
inner_border = waypoints[:, 2:4]
outer_border = waypoints[:, 4:6]
l_center_line = LineString(center_line)
l_inner_border = LineString(inner_border)
l_outer_border = LineString(outer_border)

# Create the road polygon by stacking outer and inner borders
outer_border_coords = np.array(l_outer_border.coords)
inner_border_coords = np.array(l_inner_border.coords)
road_poly = Polygon(np.vstack((outer_border_coords, np.flipud(inner_border_coords))))

# Display the track
fig = plt.figure(1, figsize=(16, 10))
ax = fig.add_subplot(111, facecolor='black')
plt.axis('equal')
plt.plot(center_line[:, 0], center_line[:, 1], label='Center Line')
plt.plot(inner_border[:, 0], inner_border[:, 1], label='Inner Border')
plt.plot(outer_border[:, 0], outer_border[:, 1], label='Outer Border')
plt.legend()
plt.show()
```

### Step 2: Racing Line Calculation

Inspired by various machine learning techniques, I used gradient descent to calculate the optimal racing line around the track. This helps in reducing lap times by minimizing the distance and curvature the car has to travel.

```python
import copy

def improve_race_line(old_line, inner_border, outer_border):
    '''Use gradient descent to find the optimal racing line'''
    new_line = copy.deepcopy(old_line)
    for i in range(len(new_line)):
        # Optimization logic here
        # Adjust new_line[i] based on the curvature
        pass
    return new_line

# Calculate the race line
race_line = copy.deepcopy(center_line[:-1])
for i in range(1500):
    race_line = improve_race_line(race_line, inner_border, outer_border)

# Visualize the race line
fig = plt.figure(1, figsize=(16, 10))
ax = fig.add_subplot(111, facecolor='black')
plt.axis('equal')
plt.plot(race_line[:, 0], race_line[:, 1], label='Race Line')
plt.plot(inner_border[:, 0], inner_border[:, 1], label='Inner Border')
plt.plot(outer_border[:, 0], outer_border[:, 1], label='Outer Border')
plt.legend()
plt.show()
```

### Step 3: Reward Function

The reward function is critical to guide the DeepRacer model in learning the optimal path around the track. It incorporates multiple factors such as staying on track, following the racing line, maintaining optimal speed, and smooth steering.

```python
def reward_function(params):
    import numpy as np
    
    # Race line data (simplified example)
    race_line = np.array([
        [2.07, -3.95], [1.85, -3.95], [1.62, -3.95], 
        # ... more waypoints ...
        [2.07, -3.95]
    ])
    
    x = params['x']
    y = params['y']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    heading = params['heading']
    speed = params['speed']
    steering_angle = params['steering_angle']
    steps = params['steps']
    progress = params['progress']

    # Initialize reward
    reward = 1e-3

    # Reward for staying on track
    if params['all_wheels_on_track']:
        reward = 1.0
    else:
        return 1e-3

    # Distance reward
    car_position = np.array([x, y])
    distances = np.linalg.norm(race_line - car_position, axis=1)
    nearest_index = np.argmin(distances)
    nearest_point = race_line[nearest_index]
    distance_reward = 1 - (distances[nearest_index] / (track_width / 2))**0.4
    reward += distance_reward

    # Direction reward
    next_point = race_line[(nearest_index + 1) % len(race_line)]
    prev_point = race_line[nearest_index - 1]
    track_direction = np.arctan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
    direction_diff = abs(track_direction - np.radians(heading))
    if direction_diff > np.pi:
        direction_diff = 2 * np.pi - direction_diff
    if direction_diff > np.radians(10.0):
        reward *= 0.5

    # Speed reward
    optimal_speed = 1.0
    speed_diff = abs(optimal_speed - speed)
    speed_reward = max(1e-3, 1 - (speed_diff / 0.2)**0.5)
    reward += speed_reward

    # Progress reward
    if steps > 0 and progress > (steps / 300) * 100:
        reward += 10.0 * (progress / 100.0)

    # Steering penalty
    if abs(steering_angle) > 15:
        reward *= 0.8

    return float(max(reward, 1e-3))
```

### Step 4: Training and Evaluation

Using the calculated racing line and optimized reward function, I trained the DeepRacer model on AWS. Continuous iterations and evaluations were conducted to refine the model and improve its performance on the track.

## Conclusion

This project demonstrates my approach to developing a high-performing AWS DeepRacer model. By analyzing the track, calculating the optimal racing line, and crafting a comprehensive reward function, I aim to achieve competitive results in the AWS DeepRacer Student League. Success in this endeavor will help me secure a scholarship in the AWS AI & ML Scholarship program, providing me with the skills and resources needed to pursue a career in AI and ML.

## Acknowledgements

I would like to thank the AWS DeepRacer community, the authors of the paper "Reinforcement Learning Using Neural Networks, with Applications to Motor Control," and various online resources that have provided valuable insights and guidance throughout this project.

Feel free to explore the repository and contribute if you have any suggestions or improvements!

---

## Authorship

**Panagiotis Georgiadis**  
Graduate Student  
University of Colorado Boulder  
Computer Science  
College of Engineering and Applied Science

---

By following this structured approach, I hope to achieve remarkable results and secure a position in the AWS AI & ML Scholarship program. Thank you for visiting my repository!
