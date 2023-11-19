from SteeringBehaviors import Wander
import SimulationEnvironment as sim

import numpy as np
import pandas as pd

def collect_training_data(total_actions):
    #set-up environment
    sim_env = sim.SimulationEnvironment()

    #robot control
    action_repeat = 100
    steering_behavior = Wander(action_repeat)

    num_params = 7
    # a single sample will be comprised of: sensor_readings, action, collision
    # Initialize network_params as an empty array with shape (total_actions, num_params)
    network_params = np.zeros((total_actions, num_params))

    for action_i in range(total_actions):
        progress = 100*float(action_i)/total_actions
        print(f'Collecting Training Data {progress}%   ', end="\r", flush=True)

        #steering_force is used for robot control only
        action, steering_force = steering_behavior.get_action(action_i, sim_env.robot.body.angle)

        for action_timestep in range(action_repeat):
            if action_timestep == 0:
                _, collision, sensor_readings = sim_env.step(steering_force)
            else:
                _, collision, _ = sim_env.step(steering_force)

            if collision:
                steering_behavior.reset_action()
                #STUDENTS NOTE: this statement only EDITS collision of PREVIOUS action
                #if current action is very new.
                if action_timestep < action_repeat * .3: #in case prior action caused collision
                    network_params[-1][-1] = collision #share collision result with prior action
                break

        # Update network_params with the collected data
        network_params[action_i, :5] = sensor_readings
        network_params[action_i, 5] = action
        network_params[action_i, 6] = collision

    # Use pandas to save the collected data as a .csv file
    df = pd.DataFrame(network_params)
    df.to_csv('submission.csv', index=False, header=False)

if __name__ == '__main__':
    total_actions = 300000
    collect_training_data(total_actions)
