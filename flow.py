import matplotlib.pyplot as plt
import numpy as np


class Car:
    def __init__(self, preferred_speed, init_node, preferred_gap, init_speed, acceleration, delay, lane, start_time):
        self.preferred_speed = preferred_speed
        self.location = init_node
        self.preferred_gap = preferred_gap
        self.speed = init_speed
        self.acceleration = acceleration
        self.decision_delay = delay
        self.current_lane = lane
        self.start_time = start_time
        self.decision_time = 0
        self.want_to_change_lanes = 0
        self.location_history = []
        self.speed_history = []

    def next_speed(self, time_step, next_car):
        next_location = next_car.location if (next_car and next_car.location) else np.inf
        next_speed = next_car.speed if next_car else np.inf
        speed_to_test = self.speed + (self.preferred_speed - self.speed) * self.acceleration * time_step
        next_speed_to_test = self.speed + (next_speed - self.speed) * self.acceleration * time_step
        if self.location is not None:
            if self.location + speed_to_test * time_step + self.preferred_gap < next_location:
                speed = speed_to_test
            elif self.location + next_speed_to_test * time_step + self.preferred_gap < next_location:
                speed = next_speed_to_test
            elif self.location + self.preferred_gap > next_location:
                speed = 0
            else:
                speed = (next_location - self.location - self.preferred_gap) / time_step
            if np.random.rand() < 0.05:
                speed = int(0.9*speed)
            self.speed_history.append(speed)
            self.speed = speed
        else:
            self.speed = 0

    def next_location(self, time_step, length, car_ahead, car_behind, max_lanes):
        if self.location is not None:
            if (max_lanes > 1) and self.want_to_change_lanes != 0:
                if self.decision_time < self.decision_delay:
                    self.decision_time += time_step
                    next_location = self.location + time_step * self.speed
                    if next_location < length:
                        self.location = next_location
                    else:
                        self.location = None
                else:
                    self.current_lane += self.want_to_change_lanes
                    self.want_to_change_lanes = 0
                    next_location = self.location
            elif max_lanes > 1:
                self.decide_to_change_lanes(car_ahead, car_behind, max_lanes)
                next_location = self.location + time_step * self.speed
                if next_location < length:
                    self.location = next_location
                else:
                    self.location = None
            else:
                self.want_to_change_lanes = 0
                next_location = self.location + time_step * self.speed
                if next_location < length:
                    self.location = next_location
                else:
                    self.location = None
            self.location_history.append(next_location)
        else:
            self.location = None

    def decide_to_change_lanes(self, car_ahead, car_behind, max_lanes):
        if car_ahead and (max_lanes > self.current_lane + 1) and (car_ahead.speed < self.preferred_speed):
            self.want_to_change_lanes = 1
            self.decision_time = 0
        elif car_behind and (car_behind.location + self.preferred_gap > self.location) and (self.current_lane > 0):
            self.want_to_change_lanes = -1
            self.decision_time = 0
        else:
            self.want_to_change_lanes = 0

    def change_lanes(self, road):
        pass

    def __str__(self):
        return "location: "+str(self.location)+", speed: "+str(self.speed)

    def __repr__(self):
        return "car at "+str(self.location)+" going "+str(self.speed)


class Road:
    def __init__(self, length, number_of_cells, max_speed, num_of_lanes):
        self.length = length
        self.number_of_cells = number_of_cells
        self.max_speed = max_speed
        self.num_of_lanes = num_of_lanes
        self.cell_size = length / number_of_cells

    def get_number_of_cells(self):
        return self.number_of_cells

    def get_max_speed(self):
        return self.max_speed

    def get_cell_by_location(self, location):
        return self.cell_size % location

    def get_max_lanes(self):
        return self.num_of_lanes


class Simulation:
    def __init__(self, road_length, arrival_rate, average_speed, average_gap, average_init_speed, average_acceleration, average_delay):
        """
        :param road_length: road length in "location" or "speed/time" units
        :param arrival_rate: fraction of iterations duting which a new car is added (i.e. 0.5=~every second step)
        :param average_speed: The center of car speed distribution
        :param average_gap: The center of preferred gap distribution
        :param average_init_speed: The average speed with which a car enters the game
        :param average_acceleration: The center of acceleration distribution
        :param average_delay:
        """
        self.road_length = road_length
        self.arrival_rate = arrival_rate
        self.average_speed = average_speed
        self.average_gap = average_gap
        self.average_init_speed = average_init_speed
        self.average_acceleration = average_acceleration
        self.average_delay = average_delay

    @staticmethod
    def update_speeds(cars, time_step):
        for num, car in enumerate(cars):
            if (len(cars) >= 1) and (num > 0):
                next_car = cars[num-1]
            else:
                next_car = None
            car.next_speed(time_step, next_car)

    @staticmethod
    def determine_cars_around(num, cars):
        if (len(cars) > 2) and (num > 0) and (num < (len(cars)-1)):
            car_ahead = cars[-num]
            car_behind = cars[-num - 2]
        elif (len(cars) > 2) and (num > 0) and (num == (len(cars)-1)):
            car_ahead = cars[-num]
            car_behind = None
        elif (len(cars) > 2) and (num == 0):
            car_ahead = None
            car_behind = cars[-num - 2]
        elif len(cars) == 2 and num == 1:
            car_ahead = None
            car_behind = cars[0]
        elif len(cars) == 2 and num == 0:
            car_ahead = cars[1]
            car_behind = None
        else:
            car_ahead = None
            car_behind = None
        return car_ahead, car_behind

    def move(self, cars, time_step, length, max_lanes, moving_lane):
        cars_to_move = []
        for num, car in enumerate(cars[moving_lane][::-1]):
            if car.location is not None:
                car_ahead, car_behind = self.find_car_by_location(cars[moving_lane], car.location)
                old_lane = car.current_lane
                car.next_location(time_step, length, car_ahead, car_behind, max_lanes)
                new_lane = car.current_lane
                if old_lane != new_lane:
                    cars_to_move.append(cars[moving_lane].index(car))
        for c2m in cars_to_move:
            car_to_be_moved = cars[moving_lane].pop(c2m)
            new_lane = car_to_be_moved.current_lane
            car_ahead, _ = self.find_car_by_location(cars[new_lane], car_to_be_moved.location)
            if car_ahead:
                new_index = cars[new_lane].index(car_ahead)
                cars[new_lane].insert(new_index,car_to_be_moved)
            else:
                cars[new_lane].append(car_to_be_moved)


    @staticmethod
    def new_car_init(average_preferred_speed, average_gap, avergage_init_speed, average_acceleration,average_delay):
        preferred_speed = max(1, np.random.poisson(average_preferred_speed))
        preferred_gap = np.random.poisson(average_gap)
        init_speed = max(1, np.random.poisson(avergage_init_speed))
        acceleration = max(1, np.random.poisson(average_acceleration))
        delay = np.random.normal(loc=average_delay, scale=0.1)
        if delay < 0.1:
            delay = 0.1
        return preferred_speed, preferred_gap, init_speed, acceleration, delay

    @staticmethod
    def find_car_by_location(cars, location):
        if len(cars) < 2:
            closest_in_front = None
            closest_behind = None
        else:
            distances = [car.location - location if ((location is not None) and (car.location is not None)) else np.inf for car in cars]
            abs_distances = [abs(i) for i in distances]
            min_abs_val = min(abs_distances)
            idx_val = distances.index(min_abs_val) if min_abs_val in distances else distances.index(-min_abs_val)
            min_val = distances[idx_val]
            if (idx_val == 0) and min_val <= 0:
                closest_behind = cars[idx_val+1]
                closest_in_front = None
            elif (idx_val == len(cars) - 1) and (min_val >= 0):
                closest_in_front = cars[idx_val - 1]
                closest_behind = None
            elif min_val > 0:
                closest_in_front = cars[idx_val]
                closest_behind = cars[idx_val+1]
            elif min_val < 0:
                closest_in_front = cars[idx_val-1]
                closest_behind = cars[idx_val]
            else:
                closest_in_front = cars[idx_val - 1]
                closest_behind = cars[idx_val + 1]
        return closest_in_front, closest_behind

    def run(self, number_of_steps, number_of_cells, max_allowed_speed, number_of_lanes, time_step):
        cars = [[] for i in range(number_of_lanes)]
        offsets = []
        time_step = time_step
        arrival_rate = self.arrival_rate
        rd = Road(self.road_length, number_of_cells, max_allowed_speed, number_of_lanes)  # lane = [0]*2000#[Car(2, 0, 0, 0, 1), 0, 0, 0, 0, Car(1, 5, 0, 0, 1), 0, 0, 0]
        # lanes = rd.construct_lane()
        for i in range(number_of_steps):

            if (np.random.rand() < arrival_rate) and (((len(cars[0])) and cars[0][-1].location) or not len(cars[0])):
                preferred_speed, preferred_gap, init_speed, acceleration, delay = self.new_car_init(
                    average_preferred_speed=self.average_speed,
                    average_gap=self.average_gap,
                    avergage_init_speed=self.average_init_speed,
                    average_acceleration=self.average_acceleration,
                    average_delay=self.average_delay)
                new_car = Car(preferred_speed, 0, preferred_gap, init_speed, acceleration, delay, 0, i)
                cars[0].append(new_car)
                # offsets.append(i)
            for j in range(number_of_lanes):
                self.update_speeds(cars[j], time_step)
                self.move(cars, time_step, rd.length, rd.get_max_lanes(), j)
        for j in range(number_of_lanes):
            fig = plt.figure()
            for num, car in enumerate(cars[j]):
                plt.plot(range(car.start_time, car.start_time + len(car.location_history)), car.location_history)
        plt.show()
        print('done')


def define_length(edge, distribution):
    pass


def define_capacity(edge, distribution):
    pass


def setup_network(size, kind, degree_distribution=None):
    pass


def calculate_path(source, target, network):
    pass


def calculate_edge_travel_time(source, target):
    pass


def pick_source(network):
    pass


def pick_destination(network):
    pass


if __name__ == "__main__":
    sim = Simulation(road_length=2000,
                     arrival_rate=0.55,
                     average_speed=15,
                     average_gap=2,
                     average_init_speed=10,
                     average_acceleration=0.5,
                     average_delay=0.5)
    sim.run(number_of_steps=5000, number_of_cells=100, max_allowed_speed=6, number_of_lanes=1, time_step=0.1)